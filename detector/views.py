from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db import transaction
from .forms import (
    FraudDetectionForm, CustomUserCreationForm, CustomAuthenticationForm, 
    BankAccountForm, BankTransactionForm
)
from .models import CustomUser, UserLoginHistory, BankAccount, BankTransaction, FailedLoginAttempt
from .ml_model import FraudModel
from .dashboard_utils import generate_dashboard_visualizations
import json
import pandas as pd
from datetime import datetime
import os
import numpy as np
import uuid
from django.db import models
import joblib
import random

# Initialize the fraud model
fraud_model = FraudModel()

# Load the fraud detection model
model = joblib.load('transaction_detector/transaction_fraud_model.pkl')

def get_client_ip(request):
    """Get client's IP address from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0].strip()
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def get_or_create_device_id(request):
    """Get or create a unique device ID that persists across sessions"""
    device_id = request.COOKIES.get('device_id')
    
    if not device_id:
        device_id = str(uuid.uuid4())
    
    return device_id

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            # Create user but don't save to database yet
            user = form.save(commit=False)
            
            # Set IP address and device ID
            user.ip_address = get_client_ip(request)
            user.device_id = get_or_create_device_id(request)
            user.last_login_ip = user.ip_address
            user.last_login_device = user.device_id
            
            # Save the user
            user.save()
            
            # Create login history entry
            UserLoginHistory.objects.create(
                user=user,
                ip_address=user.ip_address,
                device_id=user.device_id
            )
            
            # Log the user in
            login(request, user)
            messages.success(request, 'Registration successful! Welcome to our platform.')
            
            # Create response and set cookie
            response = redirect('profile')
            response.set_cookie('device_id', user.device_id, max_age=365*24*60*60)  # 1 year expiry
            return response
    else:
        form = CustomUserCreationForm()
    return render(request, 'detector/register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        
        # Get IP and device info before form validation
        ip_address = get_client_ip(request)
        device_id = get_or_create_device_id(request)
        
        if form.is_valid():
            user = form.get_user()
            
            # Update user's last login info
            user.last_login_ip = ip_address
            user.last_login_device = device_id
            user.save()
            
            # Create login history entry
            UserLoginHistory.objects.create(
                user=user,
                ip_address=user.last_login_ip,
                device_id=device_id
            )
            
            login(request, user)
            messages.success(request, 'Login successful!')
            
            # Create response and set cookie
            response = redirect('profile')
            response.set_cookie('device_id', device_id, max_age=365*24*60*60)  # 1 year expiry
            return response
        else:
            # Handle failed login attempt
            username = request.POST.get('username', '')
            
            # Record the failed login attempt
            FailedLoginAttempt.objects.create(
                username=username,
                ip_address=ip_address,
                device_id=device_id
            )
            
            # If the user exists, increment their failed login counter
            try:
                user = CustomUser.objects.get(username=username)
                user.failed_login_attempts += 1
                user.save()
            except CustomUser.DoesNotExist:
                pass
            
            messages.error(request, 'Invalid username or password.')
    else:
        form = CustomAuthenticationForm()
    return render(request, 'detector/login.html', {'form': form})

def user_logout(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('home')

@login_required
def profile(request):
    login_history = request.user.login_history.all()[:10]  # Get last 10 logins
    bank_accounts = request.user.bank_accounts.all()
    
    # Get user's transactions (both sent and received)
    sent_transactions = BankTransaction.objects.filter(sender__user=request.user).order_by('-transaction_date')[:10]
    received_transactions = BankTransaction.objects.filter(receiver__user=request.user).order_by('-transaction_date')[:10]
    
    # Get failed login attempts for display
    failed_logins = FailedLoginAttempt.objects.filter(username=request.user.username)[:10]
    failed_login_count = request.user.failed_login_attempts
    
    return render(request, 'detector/profile.html', {
        'user': request.user,
        'login_history': login_history,
        'bank_accounts': bank_accounts,
        'sent_transactions': sent_transactions,
        'received_transactions': received_transactions,
        'failed_logins': failed_logins,
        'failed_login_count': failed_login_count
    })

def home(request):
    """Render home page with fraud detection form"""
    form = FraudDetectionForm()
    return render(request, 'detector/home.html', {'form': form})

@login_required
def dashboard(request):
    """Render dashboard with visualizations"""
    try:
        # Generate dashboard visualizations
        dashboard_data = generate_dashboard_visualizations()
        
        # Check if there was an error in generating visualizations
        if 'error' in dashboard_data:
            messages.error(request, f"Error generating visualizations: {dashboard_data['error']}")
        
        return render(request, 'detector/dashboard.html', dashboard_data)
    except Exception as e:
        messages.error(request, f"Error loading dashboard: {str(e)}")
        return redirect('home')

@login_required
@csrf_exempt
def detect_fraud(request):
    """Detect fraud for a new transaction"""
    if request.method == 'POST':
        form = FraudDetectionForm(request.POST)
        if form.is_valid():
            # Prepare transaction data
            transaction_data = {
                'cc_num': form.cleaned_data['cc_num'],
                'merchant': form.cleaned_data['merchant'],
                'category': form.cleaned_data['category'],
                'amount': form.cleaned_data['amount'],
                'zip_code': form.cleaned_data['zip_code'],
                'lat': form.cleaned_data['lat'],
                'long': form.cleaned_data['long'],
                'city_pop': form.cleaned_data['city_pop'],
                'state': form.cleaned_data['state'],
                'trans_date_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                # Use average coordinates for merchant if not provided
                'merch_lat': form.cleaned_data['lat'] + np.random.uniform(-0.5, 0.5),  # Simulate nearby merchant
                'merch_long': form.cleaned_data['long'] + np.random.uniform(-0.5, 0.5),
            }
            
            # Ensure model is loaded
            if fraud_model.model is None:
                try:
                    fraud_model.load_model()
                except:
                    # Train model if it doesn't exist
                    try:
                        fraud_model.train()
                    except Exception as e:
                        return render(request, 'detector/detect.html', {
                            'form': form, 
                            'error': f"Error training model: {str(e)}"
                        })
            
            # Make prediction
            try:
                is_fraud, fraud_prob = fraud_model.predict(transaction_data)
                
                # Prepare result
                result = {
                    'is_fraud': bool(is_fraud),
                    'fraud_probability': round(float(fraud_prob) * 100, 2),
                    'transaction': transaction_data
                }
                
                return render(request, 'detector/result.html', {'result': result})
            except Exception as e:
                return render(request, 'detector/detect.html', {
                    'form': form, 
                    'error': f"Error making prediction: {str(e)}"
                })
        else:
            # Form is invalid
            return render(request, 'detector/detect.html', {'form': form, 'error': 'Invalid form data'})
    else:
        # GET request
        form = FraudDetectionForm()
        return render(request, 'detector/detect.html', {'form': form})

@login_required
@csrf_exempt
def predict(request):
    """API endpoint for fraud prediction"""
    if request.method == 'POST':
        try:
            # Parse JSON data from request
            data = json.loads(request.body)
            
            # Ensure required fields are present
            required_fields = ['cc_num', 'merchant', 'category', 'amount']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return JsonResponse({
                    'error': f"Missing required fields: {', '.join(missing_fields)}"
                }, status=400)
            
            # Add current timestamp if not provided
            if 'trans_date_time' not in data:
                data['trans_date_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
            # Fill in default values for missing fields
            if 'merch_lat' not in data and 'lat' in data:
                data['merch_lat'] = data['lat'] + np.random.uniform(-0.5, 0.5)
            if 'merch_long' not in data and 'long' in data:
                data['merch_long'] = data['long'] + np.random.uniform(-0.5, 0.5)
            
            # Ensure model is loaded
            if fraud_model.model is None:
                try:
                    fraud_model.load_model()
                except:
                    # Train model if it doesn't exist
                    fraud_model.train()
            
            # Make prediction
            is_fraud, fraud_prob = fraud_model.predict(data)
            
            # Prepare response
            response = {
                'is_fraud': bool(is_fraud),
                'fraud_probability': round(float(fraud_prob) * 100, 2),
                'risk_level': 'high' if fraud_prob > 0.7 else 'medium' if fraud_prob > 0.3 else 'low'
            }
            
            return JsonResponse(response)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        # GET request - return API usage instructions
        return JsonResponse({
            'message': 'Fraud Detection API',
            'usage': {
                'method': 'POST',
                'content_type': 'application/json',
                'required_fields': ['cc_num', 'merchant', 'category', 'amount'],
                'optional_fields': ['lat', 'long', 'city_pop', 'state', 'zip_code', 'trans_date_time']
            }
        })

@login_required
def add_bank_account(request):
    if request.method == 'POST':
        form = BankAccountForm(request.POST)
        if form.is_valid():
            bank_account = form.save(commit=False)
            bank_account.user = request.user
            bank_account.save()
            messages.success(request, 'Bank account added successfully!')
            return redirect('profile')
    else:
        form = BankAccountForm()
    
    return render(request, 'detector/bank_account_form.html', {
        'form': form,
        'title': 'Add Bank Account'
    })

@login_required
def edit_bank_account(request, pk):
    bank_account = get_object_or_404(BankAccount, pk=pk, user=request.user)
    
    if request.method == 'POST':
        form = BankAccountForm(request.POST, instance=bank_account)
        if form.is_valid():
            form.save()
            messages.success(request, 'Bank account updated successfully!')
            return redirect('profile')
    else:
        form = BankAccountForm(instance=bank_account)
    
    return render(request, 'detector/bank_account_form.html', {
        'form': form,
        'title': 'Edit Bank Account',
        'bank_account': bank_account
    })

@login_required
def delete_bank_account(request, pk):
    bank_account = get_object_or_404(BankAccount, pk=pk, user=request.user)
    if request.method == 'POST':
        bank_account.delete()
        messages.success(request, 'Bank account deleted successfully!')
    return redirect('profile')

@login_required
def set_primary_account(request, pk):
    bank_account = get_object_or_404(BankAccount, pk=pk, user=request.user)
    bank_account.is_primary = True
    bank_account.save()
    messages.success(request, f'{bank_account.bank_name} account set as primary!')
    return redirect('profile')

def get_demo_details(risk_level):
    """Generate demo IP, device ID, and location based on risk level"""
    def generate_demo_ip(level):
        if level == 'low':
            return f"103.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"
        elif level == 'medium':
            return f"{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"
        else:  # high risk
            suspicious_ips = [
                '185.156.73.54',
                '91.109.190.8',
                '103.234.220.197',
                '185.176.27.132'
            ]
            return random.choice(suspicious_ips)
    
    def generate_device_id():
        return 'DEMO-' + ''.join(random.choices('0123456789ABCDEF', k=8))
    
    def get_location(level):
        locations = {
            'low': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai'],
            'medium': ['Singapore', 'Dubai', 'Hong Kong', 'Tokyo'],
            'high': ['Lagos', 'Moscow', 'Unknown Location', 'Private Network']
        }
        return random.choice(locations[level])

    def get_risk_score(level):
        if level == 'low':
            return random.uniform(10, 40)  # 10-40% risk
        elif level == 'medium':
            return random.uniform(50, 70)  # 50-70% risk
        else:
            return random.uniform(80, 95)  # 80-95% risk
    
    return {
        'ip_address': generate_demo_ip(risk_level),
        'device_id': generate_device_id(),
        'location': get_location(risk_level),
        'risk_score': get_risk_score(risk_level)
    }

@login_required
def make_transaction(request):
    if request.method == 'POST':
        form = BankTransactionForm(request.POST)
        if form.is_valid():
            # Verify user's password
            if not authenticate(username=request.user.username, password=form.cleaned_data['password']):
                messages.error(request, 'Invalid password. Transaction cancelled.')
                return render(request, 'detector/make_transaction.html', {'form': form})
            
            # Get sender's account (primary account)
            try:
                sender_account = request.user.bank_accounts.get(is_primary=True)
            except BankAccount.DoesNotExist:
                messages.error(request, 'No primary bank account found. Please set a primary account.')
                return redirect('profile')
            
            amount = form.cleaned_data['amount']
            
            # Check if sender has sufficient balance
            if sender_account.current_balance < amount:
                messages.error(request, 'Insufficient balance for this transaction.')
                return render(request, 'detector/make_transaction.html', {'form': form})
            
            # Find receiver's account
            receiver_account = BankAccount.objects.filter(
                account_number=form.cleaned_data['receiver_account_number'],
                bank_name=form.cleaned_data['receiver_bank_name'],
                ifsc_code=form.cleaned_data['receiver_ifsc_code']
            ).first()
            
            try:
                # Create the transaction with atomic transaction to ensure data consistency
                with transaction.atomic():
                    # If receiver account doesn't exist, create it
                    if not receiver_account:
                        # Create a system user for the receiver if doesn't exist
                        receiver_user, _ = CustomUser.objects.get_or_create(
                            username=f"system_user_{form.cleaned_data['receiver_account_number']}",
                            defaults={
                                'is_active': True,
                            }
                        )
                        receiver_user.set_unusable_password()
                        receiver_user.save()
                        
                        # Create receiver's bank account
                        receiver_account = BankAccount.objects.create(
                            user=receiver_user,
                            account_holder_name=f"Account {form.cleaned_data['receiver_account_number']}",
                            account_number=form.cleaned_data['receiver_account_number'],
                            bank_name=form.cleaned_data['receiver_bank_name'],
                            ifsc_code=form.cleaned_data['receiver_ifsc_code'],
                            branch_name='System Created',
                            mobile_number='0000000000',  # Default number
                            is_primary=True,
                            current_balance=0  # Start with zero balance
                        )
                    
                    # Get transaction details based on demo mode
                    if form.cleaned_data.get('is_demo'):
                        demo_details = get_demo_details(form.cleaned_data.get('demo_risk_level', 'low'))
                        ip_address = demo_details['ip_address']
                        device_id = demo_details['device_id']
                        location = demo_details['location']
                        risk_score = demo_details['risk_score']
                        failed_attempts = 0 if form.cleaned_data.get('demo_risk_level') == 'low' else \
                                       random.randint(1, 3) if form.cleaned_data.get('demo_risk_level') == 'medium' else \
                                       random.randint(4, 10)
                    else:
                        ip_address = get_client_ip(request)
                        device_id = get_or_create_device_id(request)
                        location = sender_account.branch_name
                        failed_attempts = request.session.get('failed_login_attempts', 0)
                        # For non-demo mode, calculate risk based on amount and failed attempts
                        risk_score = min(
                            (float(amount) / 100000) * 30 +  # Amount factor (30% weight)
                            (failed_attempts * 10) +          # Failed attempts factor
                            (0 if location == sender_account.branch_name else 20),  # Location factor
                            95  # Cap at 95%
                        )
                    
                    is_fraudulent = risk_score > 75  # Consider transactions with >75% risk as fraudulent
                    
                    if is_fraudulent:
                        messages.error(request, f'Transaction blocked: High fraud risk ({risk_score:.1f}%)')
                        return render(request, 'detector/make_transaction.html', {'form': form})
                    
                    # Create transaction record with fraud detection results
                    bank_transaction = BankTransaction.objects.create(
                        sender=sender_account,
                        receiver=receiver_account,
                        receiver_account_number=form.cleaned_data['receiver_account_number'],
                        receiver_bank_name=form.cleaned_data['receiver_bank_name'],
                        receiver_ifsc_code=form.cleaned_data['receiver_ifsc_code'],
                        amount=amount,
                        remarks=form.cleaned_data.get('remarks', ''),
                        status='completed',
                        risk_percentage=risk_score,
                        is_fraudulent=is_fraudulent,
                        device_id=device_id,
                        ip_address=ip_address,
                        failed_login_attempts=failed_attempts
                    )
                    
                    # Update sender's balance
                    sender_account.current_balance = models.F('current_balance') - amount
                    sender_account.save()
                    
                    # Update receiver's balance using F() expression to prevent race conditions
                    receiver_account.current_balance = models.F('current_balance') + amount
                    receiver_account.save()
                    
                    # Refresh from database to get updated balances
                    sender_account.refresh_from_db()
                    receiver_account.refresh_from_db()
                    
                    messages.success(request, f'Transaction completed successfully! New balance: ₹{sender_account.current_balance} (Fraud Risk: {risk_score:.1f}%)')
                    return redirect('profile')
                    
            except Exception as e:
                messages.error(request, f'Transaction failed: {str(e)}')
                return render(request, 'detector/make_transaction.html', {'form': form})
    else:
        form = BankTransactionForm()
    
    return render(request, 'detector/make_transaction.html', {'form': form})

@login_required
@csrf_exempt
def get_account_holder_details(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            account_number = data.get('account_number')
            bank_name = data.get('bank_name')
            ifsc_code = data.get('ifsc_code')
            
            if not all([account_number, bank_name, ifsc_code]):
                return JsonResponse({
                    'status': 'error',
                    'message': 'Please provide all required fields'
                }, status=400)
            
            account = BankAccount.objects.filter(
                account_number=account_number,
                bank_name=bank_name,
                ifsc_code=ifsc_code
            ).first()
            
            if account:
                return JsonResponse({
                    'status': 'success',
                    'data': {
                        'account_holder_name': account.account_holder_name,
                        'bank_name': account.bank_name,
                        'ifsc_code': account.ifsc_code
                    }
                })
            else:
                return JsonResponse({
                    'status': 'not_found',
                    'message': 'Account not found. Please verify the details.'
                })
                
        except json.JSONDecodeError:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid JSON data'
            }, status=400)
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    }, status=405)

@login_required
def transaction_history(request):
    # Get all transactions for the user
    transactions = BankTransaction.objects.filter(
        sender__user=request.user
    ).select_related('sender', 'receiver').order_by('-transaction_date')
    
    # Calculate transaction statistics
    total_transactions = transactions.count()
    safe_transactions = transactions.filter(risk_percentage__lte=50).count()
    medium_risk_transactions = transactions.filter(risk_percentage__gt=50, risk_percentage__lte=75).count()
    high_risk_transactions = transactions.filter(risk_percentage__gt=75).count()
    
    context = {
        'transactions': transactions,
        'total_transactions': total_transactions,
        'safe_transactions': safe_transactions,
        'medium_risk_transactions': medium_risk_transactions,
        'high_risk_transactions': high_risk_transactions,
    }
    
    return render(request, 'detector/transaction_history.html', context)
