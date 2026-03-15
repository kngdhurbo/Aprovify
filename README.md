# Aprovify: Loan Approval Prediction System

Aprovify is an advanced, production-ready Machine Learning web application built with **Django** and **Scikit-Learn**. It allows financial institutions to instantly evaluate an applicant's credit risk and predict the probability of loan default using a highly accurate **Random Forest Classifier**.

## 🚀 Features
- **End-to-End ML Pipeline**: Seamlessly integrates data scaling (`StandardScaler`), categorical encoding (`OrdinalEncoder`), and robust modeling.
- **Real-time Predictions**: Evaluates complex financial profiles in milliseconds natively in the browser without exporting data.
- **Interactive Visualizations**: Dynamically generates responsive **Plotly** radial gauge charts to illustrate approval confidence visually.
- **Data Leakage Prevention**: Encapsulated Scikit-Learn pipelines ensure transformations are strictly fitted to training data, maintaining data integrity.
- **Modern Fintech UI/UX**: Clean, responsive frontend built with Bootstrap 5 and custom CSS (Cyan & Peach color schemes), featuring dynamic components.

## 🛠️ Technology Stack
- **Backend:** Python, Django
- **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib
- **Frontend:** HTML5, CSS3, Bootstrap 5, FontAwesome
- **Data Visualization:** Plotly Graph Objects

## 📋 Prerequisites
Make sure you have Python 3.8+ installed on your system.

## 💻 Installation & Setup

1. **Clone or Navigate to the Repository**
   Open your terminal and navigate to the project root containing `manage.py`:
   ```bash
   cd loan_system
   ```

2. **Install Dependencies**
   Install the required Python packages:
   ```bash
   pip install django pandas scikit-learn plotly numpy joblib
   ```

3. **Initialize the Database**
   Run the initial migrations to set up Django's built-in database:
   ```bash
   python manage.py migrate
   ```

4. **Train the Machine Learning Model**
   Generate the `model_robust.pkl` file by running the standalone training script. This automatically reads the dataset, processes it, trains the Random Forest, and serializes the state:
   ```bash
   python predictor_app/train_model.py
   ```
   *(Ensure you see the printed accuracy metrics and confirmation that the model was saved successfully).*

5. **Start the Development Server**
   Spin up the local Django server:
   ```bash
   python manage.py runserver 8001
   ```

6. **Access Aprovify**
   Open your browser and navigate to: [http://127.0.0.1:8001/](http://127.0.0.1:8001/)


## 🧠 Machine Learning Approach
We benchmarked several algorithms (Logistic Regression, KNN, Naive Bayes, Decision Trees, SVM) against a **Random Forest Classifier**. The Random Forest was chosen because it mitigates overfitting via bagging and handles multicollinearity robustly. It was validated using Stratified 5-Fold Cross-Validation to maintain class balance. Key determinants identified by the model include CIBIL Score, Loan Term, and Loan Amount.

## 👨‍💻 Developers
- **Dhrubojyoti Hazra** (GitHub: [@kngdhurbo](https://github.com/kngdhurbo))
- **Isheeka Mukhopadhyay** (GitHub: [@Isheeka1devs](https://github.com/Isheeka1devs))

---
*Built as a Data Science academic project for Brainware University.*
