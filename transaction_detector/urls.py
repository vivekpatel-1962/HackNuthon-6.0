from django.urls import path
from . import views

app_name = 'transaction_detector'

urlpatterns = [
    path('make-transaction/', views.make_transaction, name='make_transaction'),
    path('transaction-history/', views.transaction_history, name='transaction_history'),
] 