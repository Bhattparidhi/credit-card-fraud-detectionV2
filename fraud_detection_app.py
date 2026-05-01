import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.header-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    margin-top: 20px;
}
.center-buttons {
    display: flex;
    justify-content: center;
    gap: 20px;
}
</style>
""", unsafe_allow_html=True)

# Load model
try:
    model = joblib.load('rf_model_8features.pkl')
    scaler = joblib.load('scaler_8features.pkl')
    model_loaded = True
except:
    st.error("❌ Model files not found.")
    model_loaded = False

# Session state
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []

if 'page' not in st.session_state:
    st.session_state.page = "home"

# TITLE
st.markdown("<div class='header-title'>💳 Credit Card Fraud Detection System</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# BUTTON NAVIGATION
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔎 Analyze Transaction", use_container_width=True):
        st.session_state.page = "analyze"

with col2:
    if st.button("📊 Analytics", use_container_width=True):
        st.session_state.page = "analytics"

with col3:
    if st.button("⚙️ Settings", use_container_width=True):
        st.session_state.page = "settings"

with col4:
    if st.button("📋 Quick Stats", use_container_width=True):
        st.session_state.page = "stats"

st.markdown("---")
model_features = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'type_CASH_OUT', 'type_TRANSFER'
]
# ================= ANALYZE =================
if st.session_state.page == "analyze":

    st.subheader("🔎 Analyze Transaction")

    st.markdown("### 💰 Enter Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        step = st.number_input("⏱️ Step (time interval)", min_value=1, max_value=500, value=10, 
                               help="Time interval in the dataset (1-500)")
        amount = st.number_input("💵 Amount", min_value=0.0, value=2000000.0,
                                help="Transaction amount in currency units")
        oldbalanceOrg = st.number_input("🏦 Old Balance (Sender)", min_value=0.0, value=10.0,
                                       help="Sender's balance before transaction")
        newbalanceOrig = st.number_input("🏦 New Balance (Sender)", min_value=0.0, value=0.0,
                                        help="Sender's balance after transaction")
    
    with col2:
        oldbalanceDest = st.number_input("🏦 Old Balance (Receiver)", min_value=0.0, value=0.0,
                                        help="Receiver's balance before transaction")
        newbalanceDest = st.number_input("🏦 New Balance (Receiver)", min_value=0.0, value=2000000.0,
                                        help="Receiver's balance after transaction")
        transaction_type = st.selectbox("📤 Transaction Type", 
                                       ["CASH_OUT", "TRANSFER", "DEBIT", "PAYMENT"],
                                       help="Type of transaction")
        
        # Additional features for risk calculation
        is_weekend = st.checkbox("📅 Weekend Transaction", value=False,
                                help="Is this transaction happening on weekend?")
        previous_fraud_count = st.number_input("🚨 Previous Frauds (Account)", min_value=0, value=0,
                                              help="Number of previous fraud attempts on this account")
    
    # Calculate additional features
    balance_change_sender = abs(newbalanceOrig - oldbalanceOrg)
    balance_change_receiver = abs(newbalanceDest - oldbalanceDest)
    
    # Risk indicators
    high_amount = amount > 1000000
    zero_balance_sender = newbalanceOrig == 0
    high_velocity = step < 50
    
    if st.button("🔍 Analyze Transaction", use_container_width=True):
        # Prepare data for model
        type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0
        type_TRANSFER = 1 if transaction_type == "TRANSFER" else 0
        
        data = pd.DataFrame([{
            'step': step,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest,
            'type_CASH_OUT': type_CASH_OUT,
            'type_TRANSFER': type_TRANSFER
        }])
        
        data = data[model_features]
        data_scaled = scaler.transform(data)
        
        # Get predictions
        proba = model.predict_proba(data_scaled)[0]
        pred = model.predict(data_scaled)[0]
        fraud_probability = proba[1]
        
        # Calculate risk score (0-100)
        risk_score = fraud_probability * 100
        
        # Add risk factors
        if high_amount:
            risk_score += 10
        if zero_balance_sender:
            risk_score += 15
        if high_velocity:
            risk_score += 8
        if previous_fraud_count > 0:
            risk_score += (5 * min(previous_fraud_count, 5))
        if is_weekend:
            risk_score += 5
        
        risk_score = min(risk_score, 100)  # Cap at 100
        
        # Determine fraud threshold (0.15 for demo)
        fraud_threshold = 0.15
        is_fraud = fraud_probability >= fraud_threshold or risk_score > 50
        
        # Store in history
        st.session_state.transaction_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'amount': amount,
            'type': transaction_type,
            'fraud_probability': fraud_probability,
            'risk_score': risk_score,
            'prediction': 1 if is_fraud else 0
        })
        
        st.markdown("---")
        
        # Display result with dramatic styling
        if is_fraud:
            st.markdown(f"""
            <div class='fraud-warning fraud-alert'>
                ⚠️ FRAUDULENT TRANSACTION DETECTED
            </div>
            """, unsafe_allow_html=True)
            
            # Play alert sound (browser based)
            st.markdown("""
            <audio autoplay>
                <source src="data:audio/wav;base64,UklGRiYAAABXQVZFZm10IBAAAAABAAEAQB8AAAB9AAACABAAZGF0YQIAAAAAAA==" type="audio/wav">
            </audio>
            """, unsafe_allow_html=True)
            
            st.error("🚨 This transaction has been flagged as FRAUDULENT!")
            
        else:
            st.markdown(f"""
            <div class='legitimate-success'>
                ✅ LEGITIMATE TRANSACTION
            </div>
            """, unsafe_allow_html=True)
            st.success("✓ This transaction appears to be legitimate.")
        
        st.markdown("---")
        
        # Detailed results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Model Confidence", f"{fraud_probability:.1%}", 
                     delta=f"{(fraud_probability - 0.5):.1%}" if fraud_probability > 0.5 else None)
        
        with col2:
            st.metric("📊 Risk Score", f"{risk_score:.1f}/100", 
                     delta=f"{risk_score - 50:.1f}" if risk_score > 50 else None)
        
        with col3:
            status = "🚨 FRAUD" if is_fraud else "✅ SAFE"
            st.metric("Verdict", status)
        
        st.markdown("---")
        
        # Risk factors breakdown
        st.markdown("### 📋 Risk Factors Breakdown")
        
        risk_factors = []
        if high_amount:
            risk_factors.append(("💰 High Amount", "Amount exceeds 1M threshold", "⚠️"))
        if zero_balance_sender:
            risk_factors.append(("0️⃣ Zero Balance Alert", "Sender's balance becomes 0 after transaction", "🔴"))
        if high_velocity:
            risk_factors.append(("⚡ High Velocity", "Transaction occurs in early time steps", "⚠️"))
        if previous_fraud_count > 0:
            risk_factors.append(("🚨 Account History", f"{previous_fraud_count} previous fraud(s)", "🔴"))
        if is_weekend:
            risk_factors.append(("📅 Weekend Activity", "Transaction on weekend", "⚠️"))
        
        if not risk_factors:
            st.info("✓ No significant risk factors detected")
        else:
            for factor_name, description, severity in risk_factors:
                st.warning(f"{severity} **{factor_name}**: {description}")
        
        st.markdown("---")
        
        # Additional insights
        st.markdown("### 🔍 Transaction Insights")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sender Balance Change", f"₹{balance_change_sender:,.0f}")
        with col2:
            st.metric("Receiver Balance Change", f"₹{balance_change_receiver:,.0f}")
        with col3:
            st.metric("Transaction Step", f"{step}")
        with col4:
            st.metric("Amount Category", "High" if high_amount else "Normal")


# ================= ANALYTICS =================
elif st.session_state.page == "analytics":

    st.markdown("### 📊 Transaction Analytics Dashboard")
    
    if len(st.session_state.transaction_history) == 0:
        st.info("📭 No transactions analyzed yet. Go to 'Analyze Transaction' to start.")
    else:
        # Convert history to DataFrame
        df_history = pd.DataFrame(st.session_state.transaction_history)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyzed", len(df_history), 
                     help="Total number of transactions analyzed")
        with col2:
            fraud_count = (df_history['prediction'] == 1).sum()
            st.metric("Fraudulent", fraud_count,
                     help="Number of fraudulent transactions detected")
        with col3:
            fraud_rate = (fraud_count / len(df_history) * 100) if len(df_history) > 0 else 0
            st.metric("Fraud Rate", f"{fraud_rate:.1f}%",
                     help="Percentage of fraudulent transactions")
        with col4:
            avg_risk = df_history['risk_score'].mean()
            st.metric("Avg Risk Score", f"{avg_risk:.1f}/100")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk score distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df_history['risk_score'], nbinsx=20, 
                                       marker_color='rgba(102, 126, 234, 0.7)',
                                       name='Risk Score'))
            fig.update_layout(
                title="📊 Risk Score Distribution",
                xaxis_title="Risk Score",
                yaxis_title="Frequency",
                hovermode='x unified',
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fraud vs Legitimate pie chart
            fraud_counts = df_history['prediction'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=['Legitimate', 'Fraudulent'],
                values=[fraud_counts.get(0, 0), fraud_counts.get(1, 0)],
                marker=dict(colors=['#2ecc71', '#e74c3c'])
            )])
            fig.update_layout(title="🥧 Transaction Classification", template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Amount analysis
        fig = go.Figure()
        for pred, label, color in [(0, 'Legitimate', '#2ecc71'), (1, 'Fraudulent', '#e74c3c')]:
            subset = df_history[df_history['prediction'] == pred]
            if len(subset) > 0:
                fig.add_trace(go.Box(y=subset['amount'], name=label, marker_color=color))
        
        fig.update_layout(
            title="💰 Amount Distribution by Transaction Type",
            yaxis_title="Amount (₹)",
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Transaction history table
        st.markdown("### 📋 Detailed Transaction History")
        
        df_display = df_history.copy()
        df_display['Status'] = df_display['prediction'].apply(lambda x: "🚨 Fraud" if x == 1 else "✅ Safe")
        df_display['Fraud Probability'] = df_display['fraud_probability'].apply(lambda x: f"{x:.1%}")
        df_display['Risk Score'] = df_display['risk_score'].apply(lambda x: f"{x:.1f}")
        df_display['Amount'] = df_display['amount'].apply(lambda x: f"₹{x:,.0f}")
        
        st.dataframe(
            df_display[['timestamp', 'Amount', 'type', 'Fraud Probability', 'Risk Score', 'Status']],
            use_container_width=True,
            hide_index=True
        )
        
        # Download data
        csv = df_history.to_csv(index=False)
        st.download_button(
            label="📥 Download Transaction Data (CSV)",
            data=csv,
            file_name=f"fraud_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ================= SETTINGS =================
elif st.session_state.page == "settings":
    st.markdown("### ⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔔 Alert Settings")
        alert_enabled = st.checkbox("Enable Audio Alerts", value=True)
        visual_alerts = st.checkbox("Enable Visual Alerts", value=True)
        show_risk_factors = st.checkbox("Show Risk Factor Breakdown", value=True)
    
    with col2:
        st.markdown("#### 📊 Model Settings")
        fraud_threshold = st.slider("Fraud Probability Threshold", 0.0, 1.0, 0.15, 0.01)
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_detailed_analysis = st.checkbox("Show Detailed Analysis", value=True)
    
    st.markdown("---")
    
    st.markdown("#### 📝 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Model Information**
        - Type: Random Forest Classifier
        - Features: 8
        - Status: ✅ Active
        """)
    
    with col2:
        st.info(f"""
        **Session Statistics**
        - Transactions Analyzed: {len(st.session_state.transaction_history)}
        - Frauds Detected: {sum(1 for t in st.session_state.transaction_history if t['prediction'] == 1)}
        - Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Transaction History", use_container_width=True):
        st.session_state.transaction_history = []
        st.success("✓ Transaction history cleared!")
# ================= STATS =================
elif st.session_state.page == "stats":

    st.subheader("📋 Quick Stats")

    total = len(st.session_state.transaction_history)
    frauds = sum(1 for t in st.session_state.transaction_history if t['prediction'] == 1)

    col1, col2 = st.columns(2)
    col1.metric("Total Transactions", total)
    col2.metric("Frauds Detected", frauds)









