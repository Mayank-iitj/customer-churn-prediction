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
    except (AttributeError, ModuleNotFoundError) as e:
        # Model compatibility issue - attempt auto-retrain
        st.warning("⚠️ Model compatibility issue detected. Retraining model...")
        if "monotonic_cst" in str(e) or "cannot import name" in str(e) or "ModuleNotFoundError" in str(e):
            try:
                st.info("Training a new model compatible with current environment...")
                import subprocess
                result = subprocess.run([sys.executable, 'train_quick.py'], 
                                      capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    st.success("✓ Model retrained successfully! Please refresh the page.")
                    # Try loading again
                    model_path = os.path.join('models', 'best_model.pkl')
                    preprocessor_path = os.path.join('models', 'preprocessor.pkl')
                    if os.path.exists(model_path):
                        model = joblib.load(model_path)
                        preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
                        return model, preprocessor
                else:
                    st.error(f"Retraining failed: {result.stderr}")
            except Exception as train_error:
                st.error(f"Auto-retraining failed: {train_error}")
                st.info("Please run: python train_quick.py manually")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def create_input_form():
    """Create input form for customer data - Updated for new dataset format."""
    st.markdown('<p class="sub-header">Enter Customer Information</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 35)
    
    with col2:
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
    
    with col3:
        st.subheader("Usage & Support")
        usage_frequency = st.slider("Usage Frequency (days/month)", 0, 30, 15)
        support_calls = st.slider("Support Calls", 0, 20, 2)
        payment_delay = st.slider("Payment Delay (days)", 0, 30, 0)
    
    col1, col2 = st.columns(2)
    with col1:
        total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0, value=500.0, step=10.0)
    with col2:
        last_interaction = st.slider("Days Since Last Interaction", 0, 90, 10)
    
    # Create customer data dictionary matching new dataset format
    customer_data = {
        'Age': float(age),
        'Gender': gender,
        'Tenure': float(tenure),
        'Usage Frequency': float(usage_frequency),
        'Support Calls': float(support_calls),
        'Payment Delay': float(payment_delay),
        'Subscription Type': subscription_type,
        'Contract Length': contract_length,
        'Total Spend': float(total_spend),
        'Last Interaction': float(last_interaction),
    }
    
    return pd.DataFrame([customer_data])


def predict_churn(model, preprocessor, data):
    """Make churn prediction."""
    try:
        # Preprocess the data if preprocessor is available
        if preprocessor is not None:
            # Handle both dict-style and object-style preprocessors
            if isinstance(preprocessor, dict):
                # New format from train_quick.py
                data_processed = data.copy()
                
                # Encode categorical columns
                for col in preprocessor.get('categorical_cols', []):
                    if col in data_processed.columns and col in preprocessor['label_encoders']:
                        le = preprocessor['label_encoders'][col]
                        data_processed[col] = le.transform(data_processed[col].astype(str))
                
                # Scale numerical columns
                if 'scaler' in preprocessor and 'numerical_cols' in preprocessor:
                    num_cols = preprocessor['numerical_cols']
                    if num_cols:
                        data_processed[num_cols] = preprocessor['scaler'].transform(data_processed[num_cols])
            else:
                # Old format - object with transform method
                data_processed = preprocessor.transform(data)
        else:
            data_processed = data
            
        prediction = model.predict(data_processed)
        probability = model.predict_proba(data_processed)
        
        return prediction[0], probability[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        import traceback
        st.error(traceback.format_exc())
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


def batch_prediction_interface(model, preprocessor):
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
                    # Store original df for display
                    df_original = df.copy()
                    
                    # Remove ID columns before processing
                    id_cols = ['CustomerID', 'customer_id', 'id', 'ID']
                    cols_to_drop = [col for col in id_cols if col in df.columns]
                    if cols_to_drop:
                        df = df.drop(columns=cols_to_drop)
                    
                    # Preprocess data if preprocessor is available
                    if preprocessor is not None:
                        # Handle both dict-style and object-style preprocessors
                        if isinstance(preprocessor, dict):
                            df_processed = df.copy()
                            
                            # Encode categorical columns
                            for col in preprocessor.get('categorical_cols', []):
                                if col in df_processed.columns and col in preprocessor['label_encoders']:
                                    le = preprocessor['label_encoders'][col]
                                    df_processed[col] = le.transform(df_processed[col].astype(str))
                            
                            # Scale numerical columns
                            if 'scaler' in preprocessor and 'numerical_cols' in preprocessor:
                                num_cols = preprocessor['numerical_cols']
                                if num_cols:
                                    df_processed[num_cols] = preprocessor['scaler'].transform(df_processed[num_cols])
                        else:
                            df_processed = preprocessor.transform(df)
                    else:
                        df_processed = df
                    
                    # Make predictions
                    predictions = model.predict(df_processed)
                    probabilities = model.predict_proba(df_processed)
                    
                    # Add results to original dataframe (with ID columns)
                    df_original['Churn_Prediction'] = predictions
                    df_original['Churn_Probability'] = probabilities[:, 1]
                    df_original['Risk_Level'] = df_original['Churn_Probability'].apply(
                        lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.4 else 'Low'
                    )
                    
                    # Use df_original for display
                    df = df_original
                    
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
                prediction, probability = predict_churn(model, preprocessor, customer_data)
                
                if prediction is not None:
                    display_prediction(prediction, probability)
    
    elif app_mode == "Batch Prediction":
        # Batch predictions
        batch_prediction_interface(model, preprocessor)
    
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
