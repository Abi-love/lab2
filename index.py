import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from river import drift
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.base_model = None
        self.ensemble_models = []
        self.drift_detector = drift.ADWIN()
        self.label_encoders = {}
        
    def load_and_preprocess_data(self, file_path):
        # Load data
        df = pd.read_csv(file_path)
        
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Convert categorical columns to numeric
        for column in df.select_dtypes(include=['object']).columns:
            if column != 'churn':  # Skip target variable if it's categorical
                self.label_encoders[column] = LabelEncoder()
                df[column] = self.label_encoders[column].fit_transform(df[column])
        
        return df
    
    def prepare_time_based_split(self, df):
        # Split data by sample size instead of years
        train_size = 0.7
        train_data, test_data = train_test_split(df, train_size=train_size, random_state=42)
        
        # Separate features and target
        X_train = train_data.drop(['churn'], axis=1)
        y_train = train_data['churn']
        X_test = test_data.drop(['churn'], axis=1)
        y_test = test_data['churn']
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def handle_class_imbalance(self, X, y):
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        return X_balanced, y_balanced
    
    def train_base_model(self, X_train, y_train):
        # Train initial logistic regression model
        self.base_model = LogisticRegression(class_weight='balanced')
        self.base_model.fit(X_train, y_train)
    
    def train_ensemble_models(self, df):
        # Train models on different subsets
        n_splits = 3
        for i in range(n_splits):
            subset = df.sample(frac=0.8, random_state=i)
            X = subset.drop(['churn'], axis=1)
            y = subset['churn']
            
            X_scaled = self.scaler.transform(X)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            self.ensemble_models.append(model)
    
    def online_learning_update(self, X, y):
        # Online learning using SGD Classifier
        sgd_model = SGDClassifier(loss='log_loss', random_state=42)
        sgd_model.partial_fit(X, y, classes=np.unique(y))
        return sgd_model
    
    def detect_drift(self, predictions):
        for prediction in predictions:
            self.drift_detector.update(prediction)
            if self.drift_detector.drift_detected:
                return True
        return False
    
    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions),
            'auc_roc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        }
    
    def predict_churn(self, X_new):
        # Ensemble prediction using all models
        predictions = []
        for model in self.ensemble_models:
            predictions.append(model.predict_proba(X_new)[:, 1])
        
        # Average predictions with more weight to recent models
        weights = np.linspace(0.5, 1.0, len(predictions))
        weighted_predictions = np.average(predictions, weights=weights, axis=0)
        return (weighted_predictions > 0.5).astype(int)

# Usage example
if __name__ == "__main__":
    predictor = ChurnPredictor()
    
    # Load and preprocess data
    df = predictor.load_and_preprocess_data('dataSet.csv')
    
    # Prepare time-based split
    X_train, X_test, y_train, y_test = predictor.prepare_time_based_split(df)
    
    # Handle class imbalance
    X_balanced, y_balanced = predictor.handle_class_imbalance(X_train, y_train)
    
    # Train base model
    predictor.train_base_model(X_balanced, y_balanced)
    
    # Train ensemble models
    predictor.train_ensemble_models(df)
    
    # Evaluate base model
    base_metrics = predictor.evaluate_model(predictor.base_model, X_test, y_test)
    print("Base Model Metrics:", base_metrics)
    
    # Check for drift
    predictions = predictor.base_model.predict(X_test)
    if predictor.detect_drift(predictions):
        print("Concept drift detected! Updating models...")
        predictor.online_learning_update(X_test, y_test)