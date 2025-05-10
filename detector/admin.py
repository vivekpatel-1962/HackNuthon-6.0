from django.contrib import admin
from .models import CustomUser, UserLoginHistory, BankAccount, Transaction, FraudDetectionModel, BankTransaction

# Register your models here.
admin.site.register(CustomUser)
admin.site.register(UserLoginHistory)
admin.site.register(BankAccount)
admin.site.register(Transaction)
admin.site.register(FraudDetectionModel)
admin.site.register(BankTransaction)
