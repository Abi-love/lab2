
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from skmultiflow.drift_detection import ADWIN
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
def load_and_preprocess_data():
    # Simulated data loading (replace with actual data loading)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.randint(20000, 150000, 1000),
        'usage_minutes': np.random.randint(0, 1000, 1000),
        'data_usage': np.random.randint(0, 100, 1000),
        'support_tickets': np.random.randint(0, 10, 1000),
        'response_time': np.random.randint(1, 60, 1000),
        'monthly_bill': np.random.randint(30, 200, 1000),
        'outstanding_balance': np.random.randint(0, 1000, 1000),
        'churn': np.random.randint(0, 2, 1000),
        'timestamp': pd.date_range(start='2020-01-01', periods=1000, freq='D')
    })
    return df

# Handle class imbalance using SMOTE
def handle_imbalance(X, y):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced

# Time-weighted learning
def apply_time_weights(df):
    max_date = df['timestamp'].max()
    df['time_weight'] = (df['timestamp'] - df['timestamp'].min()).dt.days / \
                       (max_date - df['timestamp'].min()).dt.days
    return df

# Train initial model
def train_initial_model(X_train, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

# Train online learning model
def train_online_model(X_train, y_train):
    model = SGDClassifier(loss='log', random_state=42)
    model.partial_fit(X_train, y_train, classes=np.unique(y_train))
    return model

# Ensemble model training
def train_ensemble_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba)
    }
    return metrics

# Drift detection
def detect_drift(predictions):
    adwin = ADWIN()
    drift_points = []
    
    for i, pred in enumerate(predictions):
        adwin.add_element(pred)
        if adwin.detected_change():
            drift_points.append(i)
    
    return drift_points

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Split features and target
    X = df.drop(['churn', 'timestamp'], axis=1)
    y = df['churn']
    
    # Split data by time periods
    train_data = df[df['timestamp'] < '2022-01-01']
    test_data = df[df['timestamp'] >= '2022-01-01']
    
    X_train = train_data.drop(['churn', 'timestamp'], axis=1)
    y_train = train_data['churn']
    X_test = test_data.drop(['churn', 'timestamp'], axis=1)
    y_test = test_data['churn']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance
    X_train_balanced, y_train_balanced = handle_imbalance(X_train_scaled, y_train)
    
    # Apply time weights
    train_data = apply_time_weights(train_data)
    
    # Train models
    initial_model = train_initial_model(X_train_balanced, y_train_balanced)
    online_model = train_online_model(X_train_balanced, y_train_balanced)
    ensemble_model = train_ensemble_model(X_train_balanced, y_train_balanced)
    
    # Evaluate models
    initial_metrics = evaluate_model(initial_model, X_test_scaled, y_test)
    online_metrics = evaluate_model(online_model, X_test_scaled, y_test)
    ensemble_metrics = evaluate_model(ensemble_model, X_test_scaled, y_test)
    
    # Detect drift
    predictions = initial_model.predict(X_test_scaled)
    drift_points = detect_drift(predictions)
    
    # Print results
    print("Initial Model Metrics:", initial_metrics)
    print("Online Model Metrics:", online_metrics)
    print("Ensemble Model Metrics:", ensemble_metrics)
    print("Drift detected at points:", drift_points)

if __name__ == "__main__":
    main()
