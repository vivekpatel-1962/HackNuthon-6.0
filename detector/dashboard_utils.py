import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64

def get_transaction_stats(data):
    """Get basic statistics for transactions"""
    total_transactions = len(data)
    total_fraud = data['is_fraud'].sum()
    fraud_percentage = (total_fraud / total_transactions) * 100
    avg_transaction = data['amount'].mean()
    max_transaction = data['amount'].max()
    
    return {
        'total_transactions': total_transactions,
        'total_fraud': int(total_fraud),
        'fraud_percentage': round(fraud_percentage, 2),
        'avg_transaction': round(avg_transaction, 2),
        'max_transaction': round(max_transaction, 2)
    }

def generate_fraud_by_category(data):
    """Generate fraud by category visualization"""
    fraud_by_category = data.groupby('category')['is_fraud'].mean() * 100
    fraud_by_category = fraud_by_category.sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=fraud_by_category.index, y=fraud_by_category.values)
    plt.title('Fraud Percentage by Category')
    plt.xlabel('Category')
    plt.ylabel('Fraud Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    
    return base64.b64encode(img.getvalue()).decode()

def generate_fraud_by_amount(data):
    """Generate fraud by amount range visualization"""
    # Create amount bins
    bins = [0, 10, 50, 100, 500, 1000, data['amount'].max()]
    labels = ['0-10', '10-50', '50-100', '100-500', '500-1000', '1000+']
    data['amount_bin'] = pd.cut(data['amount'], bins=bins, labels=labels)
    
    # Calculate fraud percentage by amount bin
    fraud_by_amount = data.groupby('amount_bin')['is_fraud'].mean() * 100
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=fraud_by_amount.index, y=fraud_by_amount.values)
    plt.title('Fraud Percentage by Transaction Amount')
    plt.xlabel('Amount Range ($)')
    plt.ylabel('Fraud Percentage')
    plt.tight_layout()
    
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    
    return base64.b64encode(img.getvalue()).decode()

def generate_fraud_by_time(data):
    """Generate fraud by time of day visualization"""
    # Extract hour from transaction time
    data['hour'] = data['trans_date_trans_time'].dt.hour
    
    # Calculate fraud percentage by hour
    fraud_by_hour = data.groupby('hour')['is_fraud'].mean() * 100
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=fraud_by_hour.index, y=fraud_by_hour.values, marker='o')
    plt.title('Fraud Percentage by Hour of Day')
    plt.xlabel('Hour (24-hour format)')
    plt.ylabel('Fraud Percentage')
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    
    return base64.b64encode(img.getvalue()).decode()

def generate_transaction_heatmap(data):
    """Generate transaction heatmap by day and hour"""
    # Extract day of week and hour
    data['day_of_week'] = data['trans_date_trans_time'].dt.day_name()
    data['hour'] = data['trans_date_trans_time'].dt.hour
    
    # Order days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    data['day_of_week'] = pd.Categorical(data['day_of_week'], categories=day_order, ordered=True)
    
    # Create pivot table
    heatmap_data = pd.pivot_table(
        data=data,
        values='is_fraud',
        index='day_of_week',
        columns='hour',
        aggfunc=lambda x: np.mean(x) * 100
    ).fillna(0)
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, linewidths=0.5)
    plt.title('Fraud Percentage Heatmap by Day and Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    
    return base64.b64encode(img.getvalue()).decode()

def generate_geographic_fraud(data):
    """Generate fraud percentage by state"""
    # Group by state and calculate fraud percentage
    state_fraud = data.groupby('state')['is_fraud'].agg(['count', 'sum'])
    state_fraud['percentage'] = (state_fraud['sum'] / state_fraud['count']) * 100
    state_fraud = state_fraud.sort_values('percentage', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=state_fraud.index, y=state_fraud['percentage'])
    plt.title('Top 10 States by Fraud Percentage')
    plt.xlabel('State')
    plt.ylabel('Fraud Percentage')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    
    return base64.b64encode(img.getvalue()).decode()
    
def generate_dashboard_visualizations(data_path=None):
    """Generate all visualizations for the dashboard"""
    try:
        if data_path is None:
            data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fraudTrain_small.csv')
        
        # Load data
        data = pd.read_csv(data_path)
        data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
        
        # Handle column name differences
        if 'amt' in data.columns:
            data.rename(columns={'amt': 'amount'}, inplace=True)
        
        # Generate visualizations
        stats = get_transaction_stats(data)
        category_viz = generate_fraud_by_category(data)
        amount_viz = generate_fraud_by_amount(data)
        time_viz = generate_fraud_by_time(data)
        geo_viz = generate_geographic_fraud(data)
        heatmap_viz = generate_transaction_heatmap(data)
        
        return {
            'stats': stats,
            'category_viz': category_viz,
            'amount_viz': amount_viz,
            'time_viz': time_viz,
            'geo_viz': geo_viz,
            'heatmap_viz': heatmap_viz
        }
    except Exception as e:
        # Return empty visualizations in case of error
        return {
            'stats': {
                'total_transactions': 0,
                'total_fraud': 0,
                'fraud_percentage': 0,
                'avg_transaction': 0,
                'max_transaction': 0
            },
            'category_viz': '',
            'amount_viz': '',
            'time_viz': '',
            'geo_viz': '',
            'heatmap_viz': '',
            'error': str(e)
        } 