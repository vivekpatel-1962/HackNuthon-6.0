from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('detect/', views.detect_fraud, name='detect_fraud'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('predict/', views.predict, name='predict'),
    
    # Authentication URLs
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('profile/', views.profile, name='profile'),
    
    # Bank Account Management URLs
    path('accounts/add/', views.add_bank_account, name='add_bank_account'),
    path('accounts/<int:pk>/edit/', views.edit_bank_account, name='edit_bank_account'),
    path('accounts/<int:pk>/delete/', views.delete_bank_account, name='delete_bank_account'),
    path('accounts/<int:pk>/set-primary/', views.set_primary_account, name='set_primary_account'),
    
    # Transaction URLs
    path('transaction/new/', views.make_transaction, name='make_transaction'),
    path('transaction/verify-account/', views.get_account_holder_details, name='verify_account'),
    path('transaction-history/', views.transaction_history, name='transaction_history'),
] 