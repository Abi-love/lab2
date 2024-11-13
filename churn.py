import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')


def load_and_explore_data(file_path):
    """
    Load and explore the Excel dataset
    """
    # Load the data
    print("Loading dataset...")
    df = pd.read_excel(file_path)

    # Basic dataset exploration
    print("\nDataset Overview:")
    print(f"Shape: {df.shape}")
    print("\nColumns:", df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nData Types:")
    print(df.dtypes)

    print("\nSample of the data:")
    print(df.head())

    # Check class distribution
    if 'churn' in df.columns:
        print("\nClass Distribution (Churn):")
        print(df['churn'].value_counts(normalize=True))

    return df


def prepare_time_based_split(df, timestamp_col, train_years=2):
    """
    Split the data based on time periods:
    - First 2 years for training
    - Last year for testing
    """
    # Convert timestamp to datetime if it's not already
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    else:
        raise ValueError(f"{timestamp_col} column not found in dataset")

    # Sort by timestamp
    df = df.sort_values(timestamp_col)

    # Calculate the cutoff date for train/test split
    start_date = df[timestamp_col].min()
    train_cutoff = start_date + pd.DateOffset(years=train_years)

    # Split the data
    train_data = df[df[timestamp_col] < train_cutoff]
    test_data = df[df[timestamp_col] >= train_cutoff]

    print("\nData Split Summary:")
    print(f"Training set size: {len(train_data)} ({len(train_data) / len(df):.1%})")
    print(f"Testing set size: {len(test_data)} ({len(test_data) / len(df):.1%})")
    print(f"Training period: {start_date.date()} to {train_cutoff.date()}")
    print(f"Testing period: {train_cutoff.date()} to {df[timestamp_col].max().date()}")

    return train_data, test_data


class ChurnPredictor:
    # [Previous ChurnPredictor class code remains the same]
    # ... [Keep all the methods from the previous version]
    def __init__(self):
        self.scaler = StandardScaler()
        self.base_model = None
        self.ensemble_models = []

    def preprocess_data(self, df):
        """
        Preprocess the data including handling missing values and scaling
        """
        # Handle missing values
        df = df.copy()
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns

        # Fill numeric missing values with median
        for col in numeric_columns:
            df[col].fillna(df[col].median(), inplace=True)

        # Fill categorical missing values with mode
        for col in categorical_columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # Convert categorical variables to dummy variables
        df = pd.get_dummies(df, columns=categorical_columns)

        return df

    def handle_class_imbalance(self, X, y):
        """
        Handle class imbalance using SMOTE
        """
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        return X_balanced, y_balanced

    def train_time_weighted_model(self, X, y, timestamps):
        """
        Train a model with time-based weights (more recent data gets higher weights)
        """
        # Calculate weights based on time (more recent = higher weight)
        max_time = timestamps.max()
        weights = (timestamps - timestamps.min()) / (max_time - timestamps.min())
        weights = weights + 0.1  # Ensure older data still has some weight

        # Train logistic regression with sample weights
        model = LogisticRegression(random_state=42)
        model.fit(X, y, sample_weight=weights)
        return model

    def train_ensemble(self, X, y, timestamps, n_periods=3):
        """
        Train an ensemble of models on different time periods
        """
        # Split data into time periods
        period_length = len(X) // n_periods
        models = []

        for i in range(n_periods):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length if i < n_periods - 1 else len(X)

            # Train model for this period
            period_model = LogisticRegression(random_state=42)
            period_model.fit(X[start_idx:end_idx], y[start_idx:end_idx])
            models.append(('model_' + str(i), period_model))

        # Create voting classifier
        ensemble = VotingClassifier(estimators=models, voting='soft')
        ensemble.fit(X, y)
        return ensemble

    def train_online_model(self, X, y):
        """
        Train an online learning model using SGD
        """
        sgd_model = SGDClassifier(loss='log_loss', random_state=42)
        sgd_model.partial_fit(X, y, classes=np.unique(y))
        return sgd_model

    def evaluate_model(self, model, X, y):
        """
        Evaluate model performance using multiple metrics
        """
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc_roc': roc_auc_score(y, y_pred_proba)
        }

        return metrics

    def detect_drift(self, base_metrics, current_metrics, threshold=0.1):
        """
        Simple drift detection by comparing performance metrics
        """
        drift_detected = False
        metric_changes = {}

        for metric in base_metrics.keys():
            change = abs(base_metrics[metric] - current_metrics[metric])
            metric_changes[metric] = change
            if change > threshold:
                drift_detected = True

        return drift_detected, metric_changes

    def fit_predict(self, train_data, test_data):
        """
        Main method to train and evaluate all models
        """
        # Preprocess data
        X_train = self.preprocess_data(train_data.drop(['churn', 'timestamp'], axis=1))
        y_train = train_data['churn']
        timestamps_train = train_data['timestamp']

        X_test = self.preprocess_data(test_data.drop(['churn', 'timestamp'], axis=1))
        y_test = test_data['churn']

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train)

        # Train different models
        results = {}

        # 1. Base model (logistic regression)
        base_model = LogisticRegression(random_state=42)
        base_model.fit(X_train_balanced, y_train_balanced)
        results['base_model'] = self.evaluate_model(base_model, X_test, y_test)

        # 2. Time-weighted model
        time_weighted_model = self.train_time_weighted_model(
            X_train_balanced, y_train_balanced, timestamps_train
        )
        results['time_weighted'] = self.evaluate_model(time_weighted_model, X_test, y_test)

        # 3. Ensemble model
        ensemble_model = self.train_ensemble(
            X_train_balanced, y_train_balanced, timestamps_train
        )
        results['ensemble'] = self.evaluate_model(ensemble_model, X_test, y_test)

        # 4. Online learning model
        online_model = self.train_online_model(X_train_balanced, y_train_balanced)
        results['online'] = self.evaluate_model(online_model, X_test, y_test)

        # Detect drift
        drift_detected, metric_changes = self.detect_drift(
            results['base_model'],
            results['time_weighted']
        )

        return results, drift_detected, metric_changes

# Main execution
if __name__ == "__main__":
    # Specify your Excel file path
    file_path = "E:/Software Engineering 3/Artificial Intelligence (Mr Fabrice)/synthetic_telecom_churn_dataset.xlsx"  # Replace with your actual file path

    # Load and explore the data
    df = load_and_explore_data(file_path)

    # Split data based on time
    train_data, test_data = prepare_time_based_split(df)

    # Prepare features and target for training data
    X_train = train_data.drop(['churn', 'timestamp'], axis=1)
    y_train = train_data['churn']
    timestamps_train = (train_data['timestamp'] - train_data['timestamp'].min()).dt.days

    # Prepare features and target for test data
    X_test = test_data.drop(['churn', 'timestamp'], axis=1)
    y_test = test_data['churn']
    timestamps_test = (test_data['timestamp'] - test_data['timestamp'].min()).dt.days

    # Initialize predictor
    predictor = ChurnPredictor()

    # Preprocess the data
    print("\nPreprocessing data...")
    X_train_processed = predictor.preprocess_data(X_train)
    X_test_processed = predictor.preprocess_data(X_test)

    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results = predictor.fit_predict(X_train_processed, y_train, timestamps_train)

    # Print results
    print("\nModel Performance:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")





#link to report
            #https: // docs.google.com / document / d / 1uyE4bxi0JfRkBxxCQLn9ZRqufhdjaRQa / edit?usp = sharing & ouid = 107130526134161409506 & rtpof = true & sd = true