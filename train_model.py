"""
Enhanced Fraud Detection Model Training
With feature engineering and additional metrics
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# LOAD AND EXPLORE DATA
# ============================================================================

print("=" * 80)
print("ENHANCED FRAUD DETECTION MODEL TRAINING")
print("=" * 80)

# Load dataset
df = pd.read_csv('transactions.csv')

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"\n📋 Dataset Info:")
print(df.info())
print(f"\n📈 Dataset Statistics:")
print(df.describe())

# Check for missing values
print(f"\n🔍 Missing Values:")
print(df.isnull().sum())

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("DATA PREPROCESSING")
print("=" * 80)

# One-hot encode transaction types
df['type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype(int)
df['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)

# ============================================================================
# FEATURE ENGINEERING (ADDITIONAL FEATURES)
# ============================================================================

print("\n🔧 Creating Additional Features...")

# 1. Balance-based features
df['balance_change_sender'] = abs(df['newbalanceOrig'] - df['oldbalanceOrg'])
df['balance_change_receiver'] = abs(df['newbalanceDest'] - df['oldbalanceDest'])

# 2. Amount-based features
df['amount_to_sender_balance'] = df.apply(
    lambda row: (row['amount'] / row['oldbalanceOrg']) if row['oldbalanceOrg'] > 0 else 0, 
    axis=1
)
df['amount_to_receiver_balance'] = df.apply(
    lambda row: (row['amount'] / row['oldbalanceDest']) if row['oldbalanceDest'] > 0 else 0, 
    axis=1
)

# 3. Zero balance indicators
df['sender_zero_balance'] = (df['newbalanceOrig'] == 0).astype(int)
df['receiver_zero_balance'] = (df['newbalanceDest'] == 0).astype(int)

# 4. Large transaction indicator
median_amount = df['amount'].median()
df['high_amount'] = (df['amount'] > median_amount * 2).astype(int)

# 5. Velocity features (time-based)
df['is_early_step'] = (df['step'] < 50).astype(int)
df['is_peak_activity'] = ((df['step'] > 100) & (df['step'] < 200)).astype(int)

# 6. Receiver balance change percentage
df['receiver_balance_change_pct'] = df.apply(
    lambda row: ((row['newbalanceDest'] - row['oldbalanceDest']) / row['oldbalanceDest'] * 100) 
    if row['oldbalanceDest'] > 0 else 0, 
    axis=1
)

print("✓ Additional features created:")
print("  - balance_change_sender")
print("  - balance_change_receiver")
print("  - amount_to_sender_balance")
print("  - amount_to_receiver_balance")
print("  - sender_zero_balance")
print("  - receiver_zero_balance")
print("  - high_amount")
print("  - is_early_step")
print("  - is_peak_activity")
print("  - receiver_balance_change_pct")

# ============================================================================
# MODEL 1: 8-FEATURE MODEL (Original)
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 1: 8-FEATURE RANDOM FOREST")
print("=" * 80)

# Select 8 core features
features_8 = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
              'oldbalanceDest', 'newbalanceDest', 'type_CASH_OUT', 'type_TRANSFER']

X_8 = df[features_8]
y = df['isFraud']

# Handle any NaN values
X_8 = X_8.fillna(0)

# Split data
X_train_8, X_test_8, y_train, y_test = train_test_split(X_8, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n📊 Training set size: {X_train_8.shape[0]}")
print(f"📊 Test set size: {X_test_8.shape[0]}")
print(f"🎯 Fraud distribution in training set:")
print(y_train.value_counts(normalize=True))

# Scale features
scaler_8 = StandardScaler()
X_train_8_scaled = scaler_8.fit_transform(X_train_8)
X_test_8_scaled = scaler_8.transform(X_test_8)

# Train model
print("\n🔄 Training 8-feature Random Forest model...")
model_8 = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced',  # Handle class imbalance
    n_jobs=-1
)

model_8.fit(X_train_8_scaled, y_train)

# Evaluate
y_pred_8 = model_8.predict(X_test_8_scaled)
y_pred_proba_8 = model_8.predict_proba(X_test_8_scaled)
accuracy_8 = (y_pred_8 == y_test).mean()
auc_8 = roc_auc_score(y_test, y_pred_proba_8[:, 1])

print(f"\n✅ Model 1 Performance:")
print(f"   Accuracy: {accuracy_8:.4f}")
print(f"   AUC-ROC: {auc_8:.4f}")
print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred_8, target_names=['Legitimate', 'Fraudulent']))

# Feature importance
feature_importance_8 = pd.DataFrame({
    'Feature': features_8,
    'Importance': model_8.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n🎯 Feature Importance (8-feature model):")
print(feature_importance_8.to_string(index=False))

# Save model and scaler
joblib.dump(model_8, 'rf_model_8features.pkl')
joblib.dump(scaler_8, 'scaler_8features.pkl')
print("\n💾 8-feature model and scaler saved!")

# ============================================================================
# MODEL 2: 18-FEATURE EXTENDED MODEL
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 2: 18-FEATURE EXTENDED RANDOM FOREST (OPTIONAL)")
print("=" * 80)

features_18 = features_8 + [
    'balance_change_sender', 'balance_change_receiver',
    'amount_to_sender_balance', 'amount_to_receiver_balance',
    'sender_zero_balance', 'receiver_zero_balance',
    'high_amount', 'is_early_step', 'is_peak_activity',
    'receiver_balance_change_pct'
]

X_18 = df[features_18]
X_18 = X_18.fillna(0)

X_train_18, X_test_18, _, _ = train_test_split(X_18, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler_18 = StandardScaler()
X_train_18_scaled = scaler_18.fit_transform(X_train_18)
X_test_18_scaled = scaler_18.transform(X_test_18)

# Train model
print("\n🔄 Training 18-feature Random Forest model...")
model_18 = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

model_18.fit(X_train_18_scaled, y_train)

# Evaluate
y_pred_18 = model_18.predict(X_test_18_scaled)
y_pred_proba_18 = model_18.predict_proba(X_test_18_scaled)
accuracy_18 = (y_pred_18 == y_test).mean()
auc_18 = roc_auc_score(y_test, y_pred_proba_18[:, 1])

print(f"\n✅ Model 2 Performance:")
print(f"   Accuracy: {accuracy_18:.4f}")
print(f"   AUC-ROC: {auc_18:.4f}")
print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred_18, target_names=['Legitimate', 'Fraudulent']))

# Feature importance
feature_importance_18 = pd.DataFrame({
    'Feature': features_18,
    'Importance': model_18.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n🎯 Feature Importance (18-feature model):")
print(feature_importance_18.to_string(index=False))

# Save model and scaler
joblib.dump(model_18, 'rf_model_18features.pkl')
joblib.dump(scaler_18, 'scaler_18features.pkl')
joblib.dump(features_18, 'features_18.pkl')
print("\n💾 18-feature model and scaler saved!")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Model Comparison
ax = axes[0, 0]
models = ['8-Feature', '18-Feature']
accuracies = [accuracy_8, accuracy_18]
aucs = [auc_8, auc_18]

x = np.arange(len(models))
width = 0.35

ax.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
ax.bar(x + width/2, aucs, width, label='AUC-ROC', color='lightcoral')
ax.set_ylabel('Score')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim([0.8, 1.0])

# 2. Feature Importance (8-feature)
ax = axes[0, 1]
top_features_8 = feature_importance_8.head(8)
ax.barh(top_features_8['Feature'], top_features_8['Importance'], color='skyblue')
ax.set_xlabel('Importance')
ax.set_title('Top 8 Features (8-Feature Model)')

# 3. Feature Importance (18-feature)
ax = axes[1, 0]
top_features_18 = feature_importance_18.head(10)
ax.barh(top_features_18['Feature'], top_features_18['Importance'], color='lightgreen')
ax.set_xlabel('Importance')
ax.set_title('Top 10 Features (18-Feature Model)')

# 4. Confusion Matrix (8-feature)
ax = axes[1, 1]
cm = confusion_matrix(y_test, y_pred_8)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix (8-Feature Model)')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'model_comparison.png'")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING SUMMARY REPORT")
print("=" * 80)

summary_df = pd.DataFrame({
    'Model': ['8-Feature', '18-Feature'],
    'Features': [8, 18],
    'Accuracy': [f"{accuracy_8:.4f}", f"{accuracy_18:.4f}"],
    'AUC-ROC': [f"{auc_8:.4f}", f"{auc_18:.4f}"],
    'Status': ['✅ Active', '✅ Saved']
})

print("\n" + summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE")
print("=" * 80)
print("""
Files saved:
  - rf_model_8features.pkl
  - scaler_8features.pkl
  - rf_model_18features.pkl
  - scaler_18features.pkl
  - model_comparison.png

Next steps:
  1. Use 'fraud_detection_app.py' for the Streamlit interface
  2. Run: streamlit run fraud_detection_app.py
  3. Access at: http://localhost:8501
""")
