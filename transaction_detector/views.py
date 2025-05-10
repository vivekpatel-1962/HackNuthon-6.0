from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from .models import Transaction, FraudDetectionResult
import joblib
import pandas as pd
from datetime import datetime
import numpy as np
from django.contrib import messages

# Load the trained model and preprocessor
model = joblib.load('transaction_detector/transaction_fraud_model.pkl')
preprocessor = joblib.load('transaction_detector/transaction_fraud_preprocessor.pkl')

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def get_device_id(request):
    # In a real application, you would implement device fingerprinting
    # For now, we'll use user agent as a simple device identifier
    return request.META.get('HTTP_USER_AGENT', 'unknown')

@login_required
def make_transaction(request):
    if request.method == 'POST':
        try:
            amount = float(request.POST.get('amount'))
            location = request.POST.get('location')
            
            # Create transaction record
            transaction = Transaction.objects.create(
                user=request.user,
                amount=amount,
                ip_address=get_client_ip(request),
                device_id=get_device_id(request),
                location=location,
                status='PENDING'
            )
            
            # Prepare data for fraud detection
            current_hour = datetime.now().hour
            failed_attempts = request.session.get('failed_login_attempts', 0)
            
            # Create feature vector for prediction
            features = {
                'amount': [amount],
                'hour': [current_hour],
                'failed_login_attempts': [failed_attempts],
                'ip_address': [get_client_ip(request)],
                'device_id': [get_device_id(request)],
                'location': [location]
            }
            
            # Convert to DataFrame
            df = pd.DataFrame(features)
            
            # Make prediction
            risk_score = model.predict_proba(df)[0][1] * 100
            is_fraudulent = risk_score > 75  # Consider transactions with >75% risk as fraudulent
            
            # Save fraud detection result
            FraudDetectionResult.objects.create(
                transaction=transaction,
                risk_percentage=risk_score,
                is_fraudulent=is_fraudulent,
                failed_login_attempts=failed_attempts
            )
            
            # Update transaction status
            transaction.status = 'SUCCESS' if not is_fraudulent else 'FAILED'
            transaction.save()
            
            if is_fraudulent:
                messages.error(request, f'Transaction blocked: High fraud risk ({risk_score:.1f}%)')
                return redirect('transaction_history')
            
            messages.success(request, f'Transaction successful! Fraud risk: {risk_score:.1f}%')
            return redirect('transaction_history')
            
        except Exception as e:
            messages.error(request, f'Transaction failed: {str(e)}')
            return redirect('make_transaction')
    
    return render(request, 'transaction_detector/make_transaction.html')

@login_required
def transaction_history(request):
    transactions = Transaction.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'transaction_detector/transaction_history.html', {
        'transactions': transactions
    })
