#!/usr/bin/env python3
"""
This script trains a machine learning model to predict trading signals.

It performs the following steps:
1. Loads the feature-engineered data from 'training_data.csv'.
2. Splits the data into training and testing sets.
3. Trains an XGBoost classifier, a powerful gradient boosting model.
4. Evaluates the model's performance using metrics like accuracy, precision, and recall.
5. Saves the trained model to a file ('trading_model.pkl') for live prediction.
"""
import os
import sys
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# Define the path for the data and the output model
DATA_PATH = os.path.join(os.path.dirname(__file__), 'training_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'trading_model.pkl')

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plots and saves a confusion matrix heatmap."""
    # Add labels=[0, 1] to ensure the matrix is always 2x2, even if one class is not predicted.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No Signal', 'Predicted Signal'],
                yticklabels=['Actual No Signal', 'Actual Signal'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    
    plot_filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'confusion_matrix.png')
    plt.savefig(plot_filename)
    print(f"Confusion matrix plot saved to '{plot_filename}'")

def plot_feature_importance(model, feature_names):
    """Plots and saves the feature importance from the trained model."""
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importance})
    # Get the top 15 most important features
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(15)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
    plt.title('Top 15 Feature Importances for AI Model', fontsize=16)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    plot_filename = os.path.join(os.path.dirname(__file__), 'feature_importance.png')
    plt.savefig(plot_filename)
    print(f"Feature importance plot saved to '{plot_filename}'")

def main():
    """Main function to run the model training and evaluation pipeline."""
    try:
        print("--- Starting Model Training Pipeline ---")

        # 1. Load Data
        if not os.path.exists(DATA_PATH):
            print(f"❌ Error: Training data file not found at '{DATA_PATH}'", file=sys.stderr)
            print("Please run 'api/prepare_training_data.py' first.", file=sys.stderr)
            sys.exit(1)
        
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded {len(df)} samples from '{DATA_PATH}'")

        # 2. Define Features (X) and Target (y)
        # The target is the last column, all others are features.
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        feature_names = X.columns.tolist()

        # Scale data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # 3. Split Data into Training and Testing Sets
        # 80% for training, 20% for testing. `stratify=y` ensures the class
        # distribution is the same in both sets, which is crucial for imbalanced data.
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

        # 4. Train the XGBoost Model
        print("\nTraining XGBoost classifier...")
        # `scale_pos_weight` helps the model handle imbalanced classes.
        # It's the ratio of negative class samples to positive class samples.
        num_positive_samples = (y_train == 1).sum()
        num_negative_samples = (y_train == 0).sum()

        if num_positive_samples == 0:
            print("❌ Error: Training data contains no positive samples (target=1). Cannot train the model.", file=sys.stderr)
            print("This is likely due to the `TARGET_RETURN_THRESHOLD` in `prepare_training_data.py` being too high for the historical data.", file=sys.stderr)
            sys.exit(1)

        scale_pos_weight = num_negative_samples / num_positive_samples
        
        # --- NEW: Hyperparameter Tuning with GridSearchCV ---
        # Define the grid of parameters to search.
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0]
        }

        # We are optimizing for 'precision' of the positive class (Signal 1)
        grid_search = GridSearchCV(
            estimator=XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=scale_pos_weight,
                random_state=42
            ),
            param_grid=param_grid,
            scoring='precision', # This is the key to optimizing for your goal!
            cv=3, # 3-fold cross-validation
            n_jobs=-1, # Use all available CPU cores
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print("\n--- Hyperparameter Tuning Complete ---")
        print(f"Best parameters found: {grid_search.best_params_}")
        print("Model training complete.")

        # 5. Evaluate the Model on the Test Set
        print("\n--- Model Performance Evaluation ---")        
        # Get predicted probabilities for the positive class (1)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        print("\n--- Precision-Recall Trade-off at Different Thresholds ---")
        print("Goal: Find a threshold that gives high Precision for 'Signal (1)'.")
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        for thresh in thresholds:
            # Classify as 1 if probability is above the threshold
            y_pred_tuned = (y_pred_proba >= thresh).astype(int)
            print(f"\n--- Evaluation at Threshold: {thresh} ---")
            
            # Note: A high precision for class 1 means fewer false alarms.
            # A high recall for class 1 means catching more potential signals.
            # Add `labels=[0, 1]` to prevent errors if a class is missing in predictions.
            print(classification_report(
                y_test, y_pred_tuned,
                target_names=['No Signal (0)', 'Signal (1)'],
                labels=[0, 1]
            ))

        # Use a default threshold for the main report and confusion matrix
        y_pred = (y_pred_proba >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nDefault Accuracy (at 0.5 threshold): {accuracy:.2%}")

        # 6. Save the Trained Model
        joblib.dump(model, MODEL_PATH)
        print(f"\n✅ Trained model saved successfully to: '{MODEL_PATH}'")
        # 7. Visualize the results
        plot_confusion_matrix(y_test, y_pred, "XGBoost (Threshold 0.5)")
        plot_feature_importance(model, feature_names)

    except Exception as e:
        print(f"\n❌ An error occurred during model training: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()