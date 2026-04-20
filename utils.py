"""
Utility Functions for Fraud Detection System
Contains helper functions for analysis, calculations, and data processing
"""

import numpy as np
import pandas as pd
from datetime import datetime
import hashlib
import json

# ============================================================================
# RISK CALCULATION UTILITIES
# ============================================================================

def calculate_z_score(value, data_series):
    """
    Calculate Z-score for statistical anomaly detection
    
    Args:
        value: The value to calculate Z-score for
        data_series: Series of comparison values
    
    Returns:
        float: Z-score value
    """
    mean = data_series.mean()
    std = data_series.std()
    
    if std == 0:
        return 0
    
    return abs((value - mean) / std)

def benford_first_digit(amount):
    """
    Check if amount's first digit follows Benford's Law
    
    Args:
        amount: Transaction amount
    
    Returns:
        dict: Expected frequency, actual digit, anomaly status
    """
    benford_freq = {
        1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097,
        5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
    }
    
    if amount == 0:
        return {"first_digit": 0, "anomaly": False, "expected_freq": 0}
    
    first_digit = int(str(int(amount))[0])
    expected_freq = benford_freq.get(first_digit, 0)
    
    # Digits 6-9 are less common (expected freq < 0.07)
    anomaly = expected_freq < 0.07
    
    return {
        "first_digit": first_digit,
        "expected_freq": expected_freq,
        "anomaly": anomaly,
        "description": "Unusual digit pattern" if anomaly else "Normal digit pattern"
    }

def calculate_balance_ratio(amount, balance):
    """
    Calculate amount to balance ratio
    
    Args:
        amount: Transaction amount
        balance: Account balance
    
    Returns:
        float: Ratio value
    """
    if balance == 0:
        return float('inf') if amount > 0 else 0
    return amount / balance

def check_money_laundering_chain(df, sender, receiver, threshold_amount=5000000):
    """
    Detect potential money laundering chains
    
    Args:
        df: DataFrame of transactions
        sender: Sender account ID
        receiver: Receiver account ID
        threshold_amount: Minimum amount to flag
    
    Returns:
        dict: Chain analysis results
    """
    # Find all transactions from receiver
    receiver_transactions = df[df['nameOrig'] == receiver]
    
    # Check if receiver quickly transfers to others
    large_transfers = receiver_transactions[receiver_transactions['amount'] > threshold_amount]
    
    return {
        "sender": sender,
        "receiver": receiver,
        "receiver_txn_count": len(receiver_transactions),
        "quick_transfers": len(large_transfers),
        "is_chain": len(large_transfers) > 2,
        "risk_level": "HIGH" if len(large_transfers) > 2 else "LOW"
    }

# ============================================================================
# ACCOUNT ANALYSIS UTILITIES
# ============================================================================

def get_account_statistics(df, account_id):
    """
    Calculate statistics for an account
    
    Args:
        df: DataFrame of transactions
        account_id: Account to analyze
    
    Returns:
        dict: Account statistics
    """
    account_txns = df[df['nameOrig'] == account_id]
    
    if len(account_txns) == 0:
        return None
    
    return {
        "account_id": account_id,
        "total_transactions": len(account_txns),
        "fraud_count": (account_txns['isFraud'] == 1).sum(),
        "fraud_rate": (account_txns['isFraud'] == 1).sum() / len(account_txns),
        "avg_amount": account_txns['amount'].mean(),
        "max_amount": account_txns['amount'].max(),
        "min_amount": account_txns['amount'].min(),
        "total_volume": account_txns['amount'].sum(),
        "unique_recipients": account_txns['namedest'].nunique()
    }

def detect_account_anomaly(current_txn, account_stats):
    """
    Detect if transaction deviates from account behavior
    
    Args:
        current_txn: Current transaction details
        account_stats: Historical account statistics
    
    Returns:
        dict: Anomaly indicators
    """
    if not account_stats:
        return {"anomaly_detected": False, "risk_increase": 0}
    
    anomalies = []
    risk_increase = 0
    
    # Amount anomaly (> 2x average)
    if current_txn['amount'] > account_stats['avg_amount'] * 2:
        anomalies.append("Unusual amount")
        risk_increase += 15
    
    # Amount exceeds account max
    if current_txn['amount'] > account_stats['max_amount'] * 1.5:
        anomalies.append("Amount exceeds historical maximum")
        risk_increase += 20
    
    # New recipient
    account_recipients = ['C' + str(i) for i in range(1000)]  # Placeholder
    if current_txn['namedest'] not in account_recipients:
        anomalies.append("Unknown recipient")
        risk_increase += 12
    
    return {
        "anomaly_detected": len(anomalies) > 0,
        "anomalies": anomalies,
        "risk_increase": min(risk_increase, 50)
    }

# ============================================================================
# TIME-BASED ANALYSIS UTILITIES
# ============================================================================

def get_time_features(step):
    """
    Extract time-based features from step value
    
    Args:
        step: Time step value
    
    Returns:
        dict: Time features
    """
    day = step // 24
    hour = step % 24
    is_weekend = day % 7 >= 5
    is_night = hour >= 22 or hour <= 5
    is_morning = 5 <= hour < 12
    is_afternoon = 12 <= hour < 17
    is_evening = 17 <= hour < 22
    
    return {
        "day": day,
        "hour": hour,
        "is_weekend": is_weekend,
        "is_night": is_night,
        "is_morning": is_morning,
        "is_afternoon": is_afternoon,
        "is_evening": is_evening,
        "time_period": "Night" if is_night else ("Morning" if is_morning else 
                       ("Afternoon" if is_afternoon else "Evening"))
    }

def detect_burst_activity(df, account_id, current_step, time_window=5):
    """
    Detect burst activity patterns
    
    Args:
        df: Transaction DataFrame
        account_id: Account to check
        current_step: Current time step
        time_window: Window to check (in steps)
    
    Returns:
        dict: Burst activity indicators
    """
    recent_txns = df[(df['nameOrig'] == account_id) &
                     (df['step'] > current_step - time_window) &
                     (df['step'] <= current_step)]
    
    txn_count = len(recent_txns)
    total_amount = recent_txns['amount'].sum() if txn_count > 0 else 0
    
    return {
        "recent_txn_count": txn_count,
        "total_amount": total_amount,
        "avg_amount": total_amount / txn_count if txn_count > 0 else 0,
        "is_burst": txn_count >= 5,
        "burst_severity": "Critical" if txn_count >= 10 else ("High" if txn_count >= 5 else "Normal")
    }

# ============================================================================
# RECIPIENT ANALYSIS UTILITIES
# ============================================================================

def analyze_recipient_risk(df, recipient_id):
    """
    Analyze risk profile of a recipient
    
    Args:
        df: Transaction DataFrame
        recipient_id: Recipient account ID
    
    Returns:
        dict: Recipient risk analysis
    """
    recipient_txns = df[df['namedest'] == recipient_id]
    
    if len(recipient_txns) == 0:
        return {
            "recipient_id": recipient_id,
            "status": "unknown",
            "txn_count": 0,
            "risk_level": "UNKNOWN"
        }
    
    fraud_count = (recipient_txns['isFraud'] == 1).sum()
    fraud_rate = fraud_count / len(recipient_txns)
    
    risk_level = "CRITICAL" if fraud_rate > 0.3 else ("HIGH" if fraud_rate > 0.1 else "MEDIUM" if fraud_rate > 0.01 else "LOW")
    
    return {
        "recipient_id": recipient_id,
        "txn_count": len(recipient_txns),
        "fraud_count": fraud_count,
        "fraud_rate": fraud_rate,
        "avg_incoming_amount": recipient_txns['amount'].mean(),
        "risk_level": risk_level,
        "is_suspicious": fraud_rate > 0.1
    }

# ============================================================================
# NETWORK ANALYSIS UTILITIES
# ============================================================================

def detect_circular_transactions(df, sender, receiver, depth=3):
    """
    Detect if money flows back to sender (circular transactions)
    Indicator of potential money laundering
    
    Args:
        df: Transaction DataFrame
        sender: Starting sender
        receiver: Starting receiver
        depth: How many hops to check
    
    Returns:
        dict: Circular transaction indicators
    """
    visited = {sender}
    current_node = receiver
    
    for _ in range(depth):
        # Find next receivers from current node
        next_txns = df[df['nameOrig'] == current_node]
        
        if len(next_txns) == 0:
            break
        
        # Check if money comes back
        if sender in next_txns['namedest'].values:
            return {
                "circular": True,
                "depth": _,
                "risk": "CRITICAL",
                "description": f"Circular flow detected at depth {_}"
            }
        
        # Move to next node
        next_node = next_txns['namedest'].iloc[0]
        if next_node in visited:
            break
        
        visited.add(next_node)
        current_node = next_node
    
    return {
        "circular": False,
        "depth": depth,
        "risk": "LOW",
        "description": "No circular transactions detected"
    }

# ============================================================================
# DATA HASHING UTILITIES
# ============================================================================

def create_device_hash(browser, os, ip_address):
    """
    Create a unique device fingerprint hash
    
    Args:
        browser: Browser info
        os: Operating system
        ip_address: IP address
    
    Returns:
        str: Device hash
    """
    device_string = f"{browser}_{os}_{ip_address}".lower()
    device_hash = hashlib.sha256(device_string.encode()).hexdigest()
    return device_hash[:16]  # First 16 chars

# ============================================================================
# EXPORT UTILITIES
# ============================================================================

def export_transaction_report(transactions_list, format='csv'):
    """
    Export transaction report in specified format
    
    Args:
        transactions_list: List of transaction records
        format: 'csv', 'json', or 'xlsx'
    
    Returns:
        Union[str, bytes]: Formatted report
    """
    df = pd.DataFrame(transactions_list)
    
    if format == 'csv':
        return df.to_csv(index=False)
    elif format == 'json':
        return df.to_json(orient='records', indent=2)
    elif format == 'xlsx':
        return df.to_excel(index=False)
    else:
        return df.to_csv(index=False)

def generate_fraud_summary(transactions_list):
    """
    Generate summary statistics of transactions
    
    Args:
        transactions_list: List of transaction records
    
    Returns:
        dict: Summary statistics
    """
    df = pd.DataFrame(transactions_list)
    
    total_txns = len(df)
    fraudulent_txns = (df['prediction'] == 1).sum() if 'prediction' in df.columns else 0
    fraud_rate = fraudulent_txns / total_txns if total_txns > 0 else 0
    
    return {
        "total_transactions": total_txns,
        "fraudulent_transactions": fraudulent_txns,
        "legitimate_transactions": total_txns - fraudulent_txns,
        "fraud_rate": f"{fraud_rate * 100:.2f}%",
        "total_amount": df['amount'].sum() if 'amount' in df.columns else 0,
        "avg_risk_score": df['risk_score'].mean() if 'risk_score' in df.columns else 0,
        "report_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_transaction_data(transaction_dict):
    """
    Validate transaction data before analysis
    
    Args:
        transaction_dict: Transaction data dictionary
    
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check required fields
    required_fields = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                      'oldbalanceDest', 'newbalanceDest', 'type']
    
    for field in required_fields:
        if field not in transaction_dict:
            errors.append(f"Missing required field: {field}")
    
    # Check value ranges
    if 'step' in transaction_dict and (transaction_dict['step'] < 1 or transaction_dict['step'] > 500):
        errors.append("Step must be between 1 and 500")
    
    if 'amount' in transaction_dict and transaction_dict['amount'] < 0:
        errors.append("Amount cannot be negative")
    
    if 'type' in transaction_dict:
        valid_types = ["CASH_OUT", "TRANSFER", "DEBIT", "PAYMENT"]
        if transaction_dict['type'] not in valid_types:
            errors.append(f"Invalid transaction type. Must be one of: {', '.join(valid_types)}")
    
    return len(errors) == 0, errors

# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_currency(amount, currency='₹'):
    """Format amount as currency string"""
    return f"{currency}{amount:,.2f}"

def format_percentage(value):
    """Format value as percentage string"""
    return f"{value * 100:.1f}%"

def format_risk_level(risk_score):
    """Convert risk score to readable level"""
    if risk_score < 30:
        return "🟢 Low Risk"
    elif risk_score < 50:
        return "🟡 Medium Risk"
    elif risk_score < 70:
        return "🟠 High Risk"
    else:
        return "🔴 Critical Risk"

print("✓ Utility functions module loaded successfully")
