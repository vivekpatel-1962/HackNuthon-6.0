from django.db import models
from django.contrib.auth.models import User

class Transaction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    transaction_time = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField()
    device_id = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=[
        ('SUCCESS', 'Success'),
        ('FAILED', 'Failed'),
        ('PENDING', 'Pending')
    ], default='PENDING')
    created_at = models.DateTimeField(auto_now_add=True)

class FraudDetectionResult(models.Model):
    transaction = models.OneToOneField(Transaction, on_delete=models.CASCADE, related_name='fraud_detection')
    risk_percentage = models.FloatField()
    is_fraudulent = models.BooleanField(default=False)
    detection_time = models.DateTimeField(auto_now_add=True)
    failed_login_attempts = models.IntegerField(default=0)
    
    def __str__(self):
        return f"Fraud Detection for Transaction {self.transaction.id} - Risk: {self.risk_percentage}%"
