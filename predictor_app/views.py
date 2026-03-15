import os
import joblib
import pandas as pd
import plotly.graph_objects as go
from django.shortcuts import render
from django.conf import settings

# Load the trained model globally when the app starts
# This avoids reloading the model from disk on every single request.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_robust.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print(f"[+] Loaded ML pipeline from {MODEL_PATH}")
except Exception as e:
    print(f"[-] Could not load model. Ensure train_model.py is executed first. Error: {e}")
    model = None

def home(request):
    """Renders the Home Page"""
    return render(request, 'home.html')

def about(request):
    """Renders the About Page"""
    return render(request, 'about.html')

def contact(request):
    """Renders the Contact Page"""
    return render(request, 'contact.html')

def generate_gauge_chart(probability):
    """
    Generates a Plotly Gauge Chart showing the approval probability.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Approval Confidence (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#005f73"},  # Primary Cyan
            'steps': [
                {'range': [0, 40], 'color': "#fee2e2"},   # Red-ish
                {'range': [40, 70], 'color': "#fef08a"},  # Yellow-ish
                {'range': [70, 100], 'color': "#d1fae5"}  # Green-ish
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        autosize=True,
        width=400,
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#333", 'family': "Inter, sans-serif"},
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    # Return HTML suitable for embedding
    return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})

def predict(request):
    """
    Handles both rendering the prediction form (GET) and processing the ML prediction (POST).
    """
    if request.method == 'POST':
        if not model:
            # Fallback if model isn't loaded
            return render(request, 'result.html', {
                'error': "Model is not loaded. Please train the model first."
            })
            
        try:
            # 1. Extract data from POST request
            data = {
                'no_of_dependents': int(request.POST.get('no_of_dependents', 0)),
                'education': request.POST.get('education', 'Graduate'),
                'self_employed': request.POST.get('self_employed', 'No'),
                'income_annum': float(request.POST.get('income_annum', 0)),
                'loan_amount': float(request.POST.get('loan_amount', 0)),
                'loan_term': int(request.POST.get('loan_term', 0)),
                'cibil_score': int(request.POST.get('cibil_score', 0)),
                'residential_assets_value': float(request.POST.get('residential_assets_value', 0)),
                'commercial_assets_value': float(request.POST.get('commercial_assets_value', 0)),
                'luxury_assets_value': float(request.POST.get('luxury_assets_value', 0)),
                'bank_asset_value': float(request.POST.get('bank_asset_value', 0))
            }
            
            # 2. Convert to Pandas DataFrame (matching the structure expected by the pipeline)
            # This directly feeds into our ColumnTransformer and prevents data leakage handling unseen data
            df_input = pd.DataFrame([data])
            
            # 3. Model Prediction
            # 1 -> Approved, 0 -> Rejected
            prediction_class = model.predict(df_input)[0]
            prediction_proba = model.predict_proba(df_input)[0]
            
            # Extract probability of Approval (Class 1)
            approval_prob = prediction_proba[1] 
            
            prediction_label = "Approved" if prediction_class == 1 else "Rejected"
            
            # 4. Generate Plotly Gauge Chart
            plot_html = generate_gauge_chart(approval_prob)
            
            return render(request, 'result.html', {
                'prediction': prediction_label,
                'probability': approval_prob,
                'plot_html': plot_html
            })
            
        except Exception as e:
            return render(request, 'result.html', {
                'error': f"An error occurred during prediction: {str(e)}"
            })
            
    # GET request: render the empty form
    return render(request, 'predict.html')
