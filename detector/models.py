from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
from django.core.validators import RegexValidator

# Create your models here.

class CustomUser(AbstractUser):
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    device_id = models.CharField(max_length=100, null=True, blank=True)
    last_login_ip = models.GenericIPAddressField(null=True, blank=True)
    last_login_device = models.CharField(max_length=100, null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    # Add field to store total failed login attempts
    failed_login_attempts = models.PositiveIntegerField(default=0)

    def __str__(self):
        return self.username

class BankAccount(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='bank_accounts')
    account_holder_name = models.CharField(max_length=100)
    account_number = models.CharField(
        max_length=20,
        validators=[RegexValidator(r'^\d{9,18}$', 'Enter a valid account number (9-18 digits)')]
    )
    bank_name = models.CharField(max_length=100)
    ifsc_code = models.CharField(
        max_length=11,
        validators=[RegexValidator(r'^[A-Z]{4}0[A-Z0-9]{6}$', 'Enter a valid IFSC code')]
    )
    branch_name = models.CharField(max_length=100)
    mobile_number = models.CharField(
        max_length=10,
        validators=[RegexValidator(r'^\d{10}$', 'Enter a valid 10-digit mobile number')]
    )
    current_balance = models.DecimalField(max_digits=12, decimal_places=2, default=1000000)
    is_primary = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['user', 'account_number']
        ordering = ['-is_primary', '-created_at']

    def __str__(self):
        return f"{self.bank_name} - {self.account_number[-4:]}"

    def save(self, *args, **kwargs):
        # If this is the first account or being set as primary
        if self.is_primary:
            # Set all other accounts of this user as non-primary
            BankAccount.objects.filter(user=self.user).update(is_primary=False)
        # If this is the first account for the user, make it primary
        elif not BankAccount.objects.filter(user=self.user).exists():
            self.is_primary = True
        super().save(*args, **kwargs)

class UserLoginHistory(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='login_history')
    login_time = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField()
    device_id = models.CharField(max_length=100)
    
    class Meta:
        ordering = ['-login_time']

    def __str__(self):
        return f"{self.user.username} - {self.login_time}"

class FailedLoginAttempt(models.Model):
    username = models.CharField(max_length=150)  # Store username even if user doesn't exist
    ip_address = models.GenericIPAddressField()
    device_id = models.CharField(max_length=100)
    attempt_time = models.DateTimeField(auto_now_add=True)
    reason = models.CharField(max_length=100, default="Invalid credentials")
    
    class Meta:
        ordering = ['-attempt_time']
    
    def __str__(self):
        return f"Failed login: {self.username} - {self.attempt_time}"

class Transaction(models.Model):
    cc_num = models.CharField(max_length=20, verbose_name='Credit Card Number')
    merchant = models.CharField(max_length=100)
    category = models.CharField(max_length=50)
    amount = models.FloatField()
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    gender = models.CharField(max_length=1)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=2)
    zip_code = models.CharField(max_length=10)
    lat = models.FloatField()
    long = models.FloatField()
    city_pop = models.IntegerField()
    job = models.CharField(max_length=100)
    dob = models.DateField()
    trans_date_time = models.DateTimeField()
    is_fraud = models.BooleanField(default=False)
    
    def __str__(self):
        return f"Transaction {self.id} - {self.merchant} - ${self.amount}"

class FraudDetectionModel(models.Model):
    name = models.CharField(max_length=100)
    model_file = models.CharField(max_length=255)
    accuracy = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} - Accuracy: {self.accuracy:.2f}"

class BankTransaction(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ]
    
    sender = models.ForeignKey(BankAccount, on_delete=models.CASCADE, related_name='sent_transactions')
    receiver = models.ForeignKey(BankAccount, on_delete=models.SET_NULL, null=True, blank=True, related_name='received_transactions')
    receiver_account_number = models.CharField(max_length=20, default='')
    receiver_bank_name = models.CharField(max_length=100, default='')
    receiver_ifsc_code = models.CharField(max_length=11, default='')
    amount = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    remarks = models.CharField(max_length=200, null=True, blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    reference_id = models.CharField(max_length=50, unique=True)
    transaction_date = models.DateTimeField(auto_now_add=True)
    
    # Fraud detection fields
    risk_percentage = models.FloatField(default=0)
    is_fraudulent = models.BooleanField(default=False)
    device_id = models.CharField(max_length=255, null=True, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    failed_login_attempts = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-transaction_date']
    
    def __str__(self):
        return f"{self.sender.bank_name} -> {self.receiver_bank_name} ({self.amount})"
    
    def save(self, *args, **kwargs):
        # Generate a unique reference ID if not set
        if not self.reference_id:
            self.reference_id = f"TXN{timezone.now().strftime('%Y%m%d%H%M%S')}{self.sender.id}"
        super().save(*args, **kwargs)
