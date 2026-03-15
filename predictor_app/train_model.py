"""
Aprovify: Loan Approval Prediction System
Training Script

This script handles the complete Machine Learning pipeline for predicting loan approvals.
It loads the dataset, cleans it, splits the data, applies robust preprocessing to prevent
data leakage, trains a Random Forest classifier using Stratified K-Fold Cross-Validation,
and evaluates the model. Finally, the model is serialized into 'model_robust.pkl' for
production use in the related Django web application.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("   Aprovify: Model Training & Evaluation Pipeline")
    print("="*60)

    # 1. Load the Data
    # Define absolute path to the dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, 'loan_approval_dataset.csv')
    print(f"[*] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # 2. Data Cleaning
    print("[*] Cleaning data (stripping whitespaces and dropping 'loan_id')...")
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
        
    if 'loan_id' in df.columns:
        df = df.drop('loan_id', axis=1)

    # Use LabelEncoder for the target variable 'loan_status'
    # Approved -> 1, Rejected -> 0 based on logical alignment, but LabelEncoder orders alphabetically
    # Approved (0), Rejected (1) by default. Let's explicitly map it for clarity.
    df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
    
    # Define features and target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    # 3. Train/Test Split (70/30 with Stratification)
    # Stratification ensures that the class distribution of 'Approved' vs 'Rejected' is preserved
    # in both the training and testing sets, providing a reliable evaluation benchmark.
    print("[*] Splitting dataset into 70% Training and 30% Testing sets with Stratification...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Identify numerical and categorical columns
    numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # 4. Building the Preprocessing Pipeline
    # Why encapsulate in a Pipeline?
    # By encapsulating scaling and encoding inside a pipeline, we fundamentally prevent DATA LEAKAGE.
    # The scalers and encoders learn their parameters (mean, variance, categories) exclusively
    # from the Training data during the `.fit()` call. When applying to the Test data, they only
    # perform `.transform()`, ensuring that no Test set information leaks into the training phase.
    print("[*] Constructing Preprocessing Pipeline (StandardScaler + OrdinalEncoder)...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OrdinalEncoder(handle_unknown='error'), categorical_cols)
        ])

    # 5. Defining the Model
    # Why Random Forest?
    # Random Forest is chosen for its robustness against overfitting (via bagging) and its ability
    # to handle non-linear boundaries and multicollinearity (often present in financial data, e.g.,
    # multiple highly correlated asset values). It natively computes feature importance and does not
    # require extensive parameter tuning to achieve high accuracy.
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # 6. Stratified 5-Fold Cross-Validation
    print("[*] Running Stratified 5-Fold Cross-Validation on Training data...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"    - Cross-Validation Accuracies: {cv_scores}")
    print(f"    - Mean CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # 7. Final Model Training and Evaluation
    print("[*] Training final model on the full 70% Training set...")
    rf_pipeline.fit(X_train, y_train)

    print("[*] Evaluating model on the unseen 30% Testing set...")
    y_pred = rf_pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "="*60)
    print("   Final Evaluation Metrics (Test Set)")
    print("="*60)
    print(f"Accuracy:  {acc:.4%}")
    print(f"Precision: {prec:.4%}")
    print(f"Recall:    {rec:.4%}")
    print(f"F1 Score:  {f1:.4%}")
    print("="*60 + "\n")

    # 8. Feature Importance Extraction
    print("[*] Analyzing Feature Importance...")
    try:
        # Extract the trained Random Forest from the pipeline
        trained_rf = rf_pipeline.named_steps['classifier']
        importances = trained_rf.feature_importances_
        feature_names = numerical_cols + categorical_cols
        
        # Sort and display feature importances
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        print("Top 5 Most Important Features:")
        for idx, row in importance_df.head(5).iterrows():
            print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}")
    except Exception as e:
        print(f"Could not extract feature importance: {e}")

    # 9. Model Serialization
    # We save the entire pipeline, not just the model. This is a best practice.
    # It ensures that when inference is called securely from Django views,
    # the exact same preprocessing transformations are applied dynamically to the raw POST input.
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_robust.pkl')
    print(f"\n[*] Serializing the robust pipeline to '{save_path}'...")
    joblib.dump(rf_pipeline, save_path)
    print("[+] Model saved successfully! Training script complete.")

if __name__ == "__main__":
    main()
