"""
Streamlit App for Customer Churn Prediction

This interactive web application allows users to:
- Upload customer data
- Get churn predictions
- View prediction probabilities
- Visualize feature importance
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Set backend for Streamlit compatibility
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #ff7f0e;
        padding: 10px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-high-risk {
        color: #d62728;
        font-weight: bold;
        font-size: 24px;
    }
    .prediction-low-risk {
        color: #2ca02c;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_preprocessor():
    """Load trained model and preprocessor."""
    try:
        # Find the most recent model
        models_dir = 'models'
        
        # Check if results directory exists
        if os.path.exists('results'):
            results_dirs = [d for d in os.listdir('results') if os.path.isdir(os.path.join('results', d))]
        else:
            results_dirs = []
        
        if results_dirs:
            latest_dir = max(results_dirs)
            model_path = os.path.join('results', latest_dir, 'best_model.pkl')
            preprocessor_path = os.path.join('results', latest_dir, 'preprocessor.pkl')
        else:
            model_path = os.path.join(models_dir, 'best_model.pkl')
            preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
            return model, preprocessor
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def create_input_form():
    """Create input form for customer data."""
    st.markdown('<p class="sub-header">Enter Customer Information</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    
    with col2:
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", 
                                      "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    
    with col3:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    col1, col2 = st.columns(2)
    with col1:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.5)
    with col2:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0, step=10.0)
    
    # Create customer data dictionary
    customer_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    return pd.DataFrame([customer_data])


def predict_churn(model, data):
    """Make churn prediction."""
    try:
        prediction = model.predict(data)
        probability = model.predict_proba(data)
        
        return prediction[0], probability[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None


def display_prediction(prediction, probability):
    """Display prediction results."""
    st.markdown('<p class="sub-header">Prediction Results</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Churn Prediction", 
                 "High Risk 🔴" if prediction == 1 else "Low Risk 🟢",
                 delta=None)
    
    with col2:
        st.metric("Churn Probability", 
                 f"{probability[1]*100:.1f}%",
                 delta=None)
    
    with col3:
        st.metric("Retention Probability", 
                 f"{probability[0]*100:.1f}%",
                 delta=None)
    
    # Probability visualization
    fig, ax = plt.subplots(figsize=(8, 4))
    categories = ['No Churn', 'Churn']
    colors = ['green', 'red']
    ax.barh(categories, probability, color=colors, alpha=0.7)
    ax.set_xlabel('Probability')
    ax.set_title('Churn Prediction Probabilities')
    ax.set_xlim([0, 1])
    
    for i, v in enumerate(probability):
        ax.text(v + 0.02, i, f'{v*100:.1f}%', va='center')
    
    st.pyplot(fig)
    plt.close(fig)
    
    # Recommendations
    st.markdown('<p class="sub-header">Recommendations</p>', unsafe_allow_html=True)
    
    if prediction == 1:
        st.warning("⚠️ This customer is at high risk of churning!")
        st.markdown("""
        **Recommended Actions:**
        - Reach out proactively with retention offers
        - Offer contract upgrades or loyalty discounts
        - Investigate recent service issues or complaints
        - Provide personalized customer support
        - Consider bundled service promotions
        """)
    else:
        st.success("✅ This customer is likely to stay!")
        st.markdown("""
        **Recommended Actions:**
        - Maintain current service quality
        - Consider upselling additional services
        - Encourage referrals and positive reviews
        - Monitor satisfaction regularly
        """)


def batch_prediction_interface(model):
    """Interface for batch predictions."""
    st.markdown('<p class="sub-header">Batch Prediction</p>', unsafe_allow_html=True)
    
    st.write("Upload a CSV file with customer data for batch predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} records")
            st.dataframe(df.head())
            
            if st.button("Generate Predictions"):
                with st.spinner("Making predictions..."):
                    # Make predictions
                    predictions = model.predict(df)
                    probabilities = model.predict_proba(df)
                    
                    # Add results to dataframe
                    df['Churn_Prediction'] = predictions
                    df['Churn_Probability'] = probabilities[:, 1]
                    df['Risk_Level'] = df['Churn_Probability'].apply(
                        lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.4 else 'Low'
                    )
                    
                    st.success("Predictions completed!")
                    
                    # Display summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", len(df))
                    with col2:
                        st.metric("Predicted Churners", (predictions == 1).sum())
                    with col3:
                        churn_rate = (predictions == 1).sum() / len(df) * 100
                        st.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
                    
                    # Display results
                    st.dataframe(df)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        risk_counts = df['Risk_Level'].value_counts()
                        ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                              colors=['green', 'orange', 'red'])
                        ax.set_title('Risk Level Distribution')
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.hist(df['Churn_Probability'], bins=20, edgecolor='black', alpha=0.7)
                        ax.set_xlabel('Churn Probability')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Distribution of Churn Probabilities')
                        st.pyplot(fig)
                        plt.close(fig)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")


def main():
    """Main application."""
    
    # Header
    st.markdown('<p class="main-header">📊 Customer Churn Prediction System</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This application uses machine learning to predict customer churn and help businesses 
    implement proactive retention strategies.
    """)
    
    # Load model
    model, preprocessor = load_model_and_preprocessor()
    
    if model is None:
        st.error("⚠️ No trained model found. Please train a model first by running main.py")
        st.info("To train a model, run: `python main.py` in your terminal")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose Mode", 
                                ["Single Prediction", "Batch Prediction", "Model Info"])
    
    if app_mode == "Single Prediction":
        # Single customer prediction
        customer_data = create_input_form()
        
        if st.button("Predict Churn", type="primary"):
            with st.spinner("Making prediction..."):
                prediction, probability = predict_churn(model, customer_data)
                
                if prediction is not None:
                    display_prediction(prediction, probability)
    
    elif app_mode == "Batch Prediction":
        # Batch predictions
        batch_prediction_interface(model)
    
    else:
        # Model information
        st.markdown('<p class="sub-header">Model Information</p>', unsafe_allow_html=True)
        
        st.write("**Model Type:**", type(model).__name__)
        
        if hasattr(model, 'n_features_in_'):
            st.write("**Number of Features:**", model.n_features_in_)
        
        if hasattr(model, 'feature_importances_'):
            st.write("**Feature Importance:**")
            
            # Get feature names if available
            feature_names = [f"Feature {i}" for i in range(len(model.feature_importances_))]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'], importance_df['Importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Feature Importances')
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close(fig)
        
        st.markdown("""
        ### About This Model
        
        This churn prediction model was trained using advanced machine learning techniques including:
        - Multiple classification algorithms (Random Forest, Gradient Boosting, XGBoost, etc.)
        - Hyperparameter tuning with cross-validation
        - Feature engineering and selection
        - Class imbalance handling
        
        The model provides accurate predictions to help identify at-risk customers
        and enable proactive retention strategies.
        """)


if __name__ == "__main__":
    main()
