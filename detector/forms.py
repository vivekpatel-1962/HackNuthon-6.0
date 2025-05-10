from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import Transaction, CustomUser, BankAccount, BankTransaction
from django.core.validators import RegexValidator

class TransactionForm(forms.ModelForm):
    class Meta:
        model = Transaction
        fields = ['cc_num', 'merchant', 'category', 'amount', 'first_name', 'last_name', 'gender',
                 'city', 'state', 'zip_code', 'lat', 'long', 'city_pop', 'job', 'dob']
        widgets = {
            'dob': forms.DateInput(attrs={'type': 'date'}),
        }

class FraudDetectionForm(forms.Form):
    cc_num = forms.CharField(max_length=20, label='Credit Card Number')
    merchant = forms.CharField(max_length=100)
    category = forms.ChoiceField(choices=[
        ('shopping_pos', 'Shopping - In Person'),
        ('shopping_net', 'Shopping - Online'),
        ('food_dining', 'Food & Dining'),
        ('health_fitness', 'Health & Fitness'),
        ('entertainment', 'Entertainment'),
        ('travel', 'Travel'),
        ('grocery_pos', 'Grocery - In Person'),
        ('grocery_net', 'Grocery - Online'),
        ('gas_transport', 'Gas & Transport'),
        ('misc_pos', 'Miscellaneous - In Person'),
        ('misc_net', 'Miscellaneous - Online'),
        ('kids_pets', 'Kids & Pets'),
        ('home', 'Home'),
        ('personal_care', 'Personal Care'),
    ])
    amount = forms.FloatField(min_value=0.01)
    zip_code = forms.CharField(max_length=10)
    state = forms.CharField(max_length=2)
    city_pop = forms.IntegerField(min_value=1)
    lat = forms.FloatField()
    long = forms.FloatField()

class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = CustomUser
        fields = ('username', 'password1', 'password2')
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields:
            self.fields[field].widget.attrs.update({
                'class': 'form-control',
                'placeholder': field.replace('_', ' ').title()
            })

class CustomAuthenticationForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields:
            self.fields[field].widget.attrs.update({
                'class': 'form-control',
                'placeholder': field.replace('_', ' ').title()
            })

class BankAccountForm(forms.ModelForm):
    class Meta:
        model = BankAccount
        fields = [
            'account_holder_name',
            'account_number',
            'bank_name',
            'ifsc_code',
            'branch_name',
            'mobile_number',
            'is_primary'
        ]
        widgets = {
            'account_holder_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter name as per bank account'
            }),
            'account_number': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter account number'
            }),
            'bank_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter bank name'
            }),
            'ifsc_code': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter IFSC code'
            }),
            'branch_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter branch name'
            }),
            'mobile_number': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter 10-digit mobile number'
            }),
            'is_primary': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            })
        }

class BankTransactionForm(forms.Form):
    receiver_account_number = forms.CharField(
        max_length=20,
        validators=[RegexValidator(r'^\d{9,18}$', 'Enter a valid account number (9-18 digits)')],
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    receiver_bank_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    receiver_ifsc_code = forms.CharField(
        max_length=11,
        validators=[RegexValidator(r'^[A-Z]{4}0[A-Z0-9]{6}$', 'Enter a valid IFSC code')],
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    amount = forms.DecimalField(
        max_digits=12,
        decimal_places=2,
        min_value=0.01,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    remarks = forms.CharField(
        max_length=200,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control'}),
        help_text='Enter your login password to confirm the transaction'
    )
    is_demo = forms.BooleanField(
        required=False,
        initial=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'id': 'is_demo',
            'data-bs-toggle': 'collapse',
            'data-bs-target': '#demoOptions'
        })
    )
    demo_risk_level = forms.ChoiceField(
        required=False,
        choices=[
            ('low', 'Low Risk (Random IP from same country)'),
            ('medium', 'Medium Risk (Random IP from different country)'),
            ('high', 'High Risk (Known suspicious IP)')
        ],
        widget=forms.Select(attrs={
            'class': 'form-select',
            'id': 'demo_risk_level'
        })
    ) 