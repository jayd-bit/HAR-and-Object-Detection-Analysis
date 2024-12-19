import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
def load_har_data():
    # Load features and labels
    features = pd.read_csv('UCI HAR Dataset/train/X_train.txt', sep='\s+', header=None)
    feature_names = pd.read_csv('UCI HAR Dataset/features.txt', sep='\s+', header=None)[1]
    features.columns = feature_names
    
    # Load activity labels
    labels = pd.read_csv('UCI HAR Dataset/train/y_train.txt', header=None)[0]
    activity_labels = pd.read_csv('UCI HAR Dataset/activity_labels.txt', sep='\s+', header=None)
    activity_map = dict(zip(activity_labels[0], activity_labels[1]))
    labels = labels.map(activity_map)
    
    return features, labels

# Train and evaluate models
def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, cm, y_pred

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

# Main execution
def main():
    # 1. Load and preprocess data
    print("Loading data...")
    X, y = load_har_data()
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42)
    }
    
    # 5. Train and evaluate each model
    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        metrics, cm, _ = train_evaluate_model(
            model, X_train_scaled, X_test_scaled, y_train, y_test, name
        )
        results.append(metrics)
        plot_confusion_matrix(cm, np.unique(y), name)
    
    # 6. Compare results
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('Model')
    
    # 7. Plot comparison
    plt.figure(figsize=(12, 6))
    results_df.plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # 8. Print detailed results
    print("\nDetailed Results:")
    print(results_df.round(4))

if __name__ == "__main__":
    main()
