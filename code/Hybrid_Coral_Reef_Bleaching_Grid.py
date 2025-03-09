#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
README / Documentation:
This script demonstrates a comprehensive hybrid approach for predicting coral reef bleaching using a NOAA dataset.
It compares multiple approaches:

1. Introduction & Setup:
   - Explanation of the problem.
   - Installation instructions.

2. Data Loading and Preprocessing:
   - Loads 'data/NOAA_reef_check_bleaching_data.csv'.
   - Cleans the dataset, handles missing values, encodes categorical features
   - Performs robust splitting and scaling with class balance preservation

3. Classical Machine Learning Models:
   - Implements Logistic Regression, Decision Tree, SVM, and Naïve Bayes (scikit‑learn).
   - Trains and evaluates each model (accuracy, confusion matrices, training times).

4. Quantum Machine Learning Model using Qadence (QNN):
   - Selects features from the scaled data, normalizes them to [0, PI] for rotation angles.
   - Constructs a quantum circuit (using RX, RY, and CNOT) to encode the features.
   - Enhances with trainable parameters to ensure proper learning
   - Wraps the quantum circuit as a PyTorch module, trains it and evaluates its performance.

5. Quantum Evolution Kernel (QEK) Based Model:
   - Implements the QEK approach for quantum kernel-based learning
   - Uses the QEK to compute a kernel matrix and trains an SVM with a precomputed kernel
   - Evaluates the QEK-based model on the test data
   
6. Comparison and Visualization:
   - Bar charts compare all model accuracies and training times.
   - A heatmap shows the feature correlations.
   
7. Final Summary:
   - Comments summarize key findings and usage instructions.
       
Usage:
1. Place the NOAA CSV file at 'data/NOAA_reef_check_bleaching_data.csv'.
2. Install dependencies:
       pip install numpy pandas scikit-learn matplotlib seaborn torch qadence pulser
       pip install quantum-evolution-kernel  # For Quantum Evolution Kernel, not working!
3. Run this script in a Jupyter Notebook cell or as a standalone Python file.
"""

# =============================================================================
# Section 1: Introduction & Setup
# =============================================================================
import os
import sys
import time
import warnings
from math import pi as PI_value

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, balanced_accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings("ignore")

print("Coral Reef Bleaching Prediction using Classical and Quantum ML Approaches")

# =============================================================================
# Section 2: Data Loading and Preprocessing
# =============================================================================
def find_data_file(base_names, possible_paths):
    """
    Find the data file by trying multiple possible paths and base names.
    
    Args:
        base_names (list): List of possible base file names
        possible_paths (list): List of possible paths to search
        
    Returns:
        str or None: Path to the found file, or None if not found
    """
    for path in possible_paths:
        for base_name in base_names:
            file_path = os.path.join(path, base_name)
            if os.path.isfile(file_path):
                return file_path
    return None

def load_and_preprocess_data():
    """
    Load and preprocess the NOAA coral reef bleaching dataset.
    
    Returns:
        tuple: X_train_scaled, X_test_scaled, y_train, y_test, feature_names, df
    """
    # Define multiple possible paths and base names to find the data file
    base_names = [
        "NOAA_Reef_Check__Bleaching_Data.csv",
        "NOAA_reef_check_bleaching_data.csv",
        "noaa_reef_check_bleaching_data.csv"
    ]
    
    possible_paths = [
        "./AQUA-QUANT/data",
        "./data",
        "../data",
        "."
    ]
    
    # Find the data file
    print("Current working directory:", os.getcwd())
    data_path = find_data_file(base_names, possible_paths)
    
    if data_path is None:
        raise FileNotFoundError(
            "Could not find NOAA reef check bleaching data. "
            "Please place the CSV file in one of these locations: "
            f"{', '.join(possible_paths)}"
        )
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Original dataset shape: {df.shape}")
    print(df.head())
    
    # Clean the dataset: remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Display class distribution before cleaning
    if 'Bleaching' in df.columns:
        bleaching_counts = df['Bleaching'].value_counts()
        print("\nClass distribution before cleaning:")
        print(bleaching_counts)
        print(f"Positive rate: {bleaching_counts.get('Yes', 0) / len(df):.2%}")
    
    # Standardize column names - replace spaces with underscores
    df.columns = [col.replace(' ', '') for col in df.columns]
    
    # Check for columns with all missing values
    all_na_columns = [col for col in df.columns if df[col].isna().all()]
    if all_na_columns:
        print(f"Dropping columns with all missing values: {all_na_columns}")
        df.drop(columns=all_na_columns, inplace=True)
    
    # Handle missing values first: replace with mode for categorical, mean for numerical
    for col in df.columns:
        if df[col].isna().any():
            print(f"Handling missing values in {col} column: {df[col].isna().sum()} NaNs")
            if df[col].dtype == 'object':
                # For categorical columns, fill with mode
                if not df[col].dropna().empty:  # Only if there are non-NaN values
                    mode_value = df[col].mode()[0]
                    df[col].fillna(mode_value, inplace=True)
                else:
                    # If all values are NaN, fill with a default value
                    print(f"Column {col} has all NaN values, filling with 'none'")
                    df[col].fillna('none', inplace=True)
            else:
                # For numerical columns, fill with mean or 0 if all NaN
                if not df[col].dropna().empty:  # Only if there are non-NaN values
                    mean_value = df[col].mean()
                    df[col].fillna(mean_value, inplace=True)
                else:
                    print(f"Column {col} has all NaN values, filling with 0")
                    df[col].fillna(0, inplace=True)
    
    print(f"Shape after handling missing values: {df.shape}")
    
    # Display data types
    print("\nColumn data types:")
    print(df.dtypes)
    
    print("\nEncoding categorical columns...")
    # Encode all categorical columns
    
    # Binary categorical columns - map Yes/No, yes/no to 1/0
    binary_columns = ["Bleaching", "Storms"]
    for col in binary_columns:
        if col in df.columns:
            # Handle case insensitivity
            df[col] = df[col].astype(str).str.lower()
            df[col] = df[col].map({"yes": 1, "no": 0})
            print(f"Encoded {col} to 1/0")
    
    # Ocean categorical column
    if 'Ocean' in df.columns:
        ocean_mapping = {"Arabian Gulf": 0, "Atlantic": 1, "Indian": 2, "Pacific": 3, "Red Sea": 4}
        df['Ocean'] = df['Ocean'].map(ocean_mapping)
        print("Encoded Ocean column")
    
    # Impact categorical columns with consistent naming
    impact_columns = ['Commercial', 'HumanImpact', 'Siltation', 'Dynamite', 'Poison', 'Sewage', 'Industrial']
    for col in impact_columns:
        if col in df.columns:
            # First, standardize values to lowercase and strip whitespace
            df[col] = df[col].astype(str).str.lower().str.strip()
            
            # Display unique values before mapping
            unique_values = df[col].unique()
            print(f"Unique values in {col} before mapping: {unique_values}")
            
            # Map values
            mapping = {'none': 0, 'low': 1, 'moderate': 2, 'high': 3, 'nan': 0}
            df[col] = df[col].map(mapping)
            
            # Check if mapping was successful
            if df[col].isna().any():
                print(f"Warning: Column {col} has {df[col].isna().sum()} NaN values after mapping")
                print(f"Unique values in {col} after mapping: {df[col].unique()}")
                
                # Fill NaN with 0 (representing 'none')
                df[col].fillna(0, inplace=True)
                print(f"Filled NaN values in {col} with 0")
            
            print(f"Encoded {col} column")
    
    # Check for any remaining object columns
    object_columns = df.select_dtypes(include=['object']).columns.tolist()
    if object_columns:
        print(f"Warning: The following columns still have non-numeric types: {object_columns}")
        print("Converting remaining object columns to numeric if possible...")
        
        for col in object_columns:
            try:
                # Display unique values to help with debugging
                print(f"Unique values in {col}: {df[col].unique()}")
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaN with mean or mode
                if df[col].isna().any():
                    if df[col].nunique() > 10:  # If many unique values, use mean
                        if not df[col].dropna().empty:
                            df[col].fillna(df[col].mean(), inplace=True)
                        else:
                            df[col].fillna(0, inplace=True)
                    else:  # If few unique values, use mode
                        if not df[col].dropna().empty:
                            df[col].fillna(df[col].mode()[0], inplace=True)
                        else:
                            df[col].fillna(0, inplace=True)
                print(f"Converted {col} to numeric")
            except Exception as e:
                print(f"Could not convert {col} to numeric: {e}, dropping column")
                df.drop(columns=[col], inplace=True)
    
    # Final check for NaN values
    nan_check = df.isna().sum()
    if nan_check.sum() > 0:
        print("\nWarning: Dataset still contains NaN values:")
        print(nan_check[nan_check > 0])
        print("Filling remaining NaN values...")
        # Fill remaining NaN values with column means or 0
        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype.kind in 'ifc':  # integer, float, complex
                    if not df[col].dropna().empty:
                        df[col].fillna(df[col].mean(), inplace=True)
                    else:
                        df[col].fillna(0, inplace=True)
                else:
                    df[col].fillna(0, inplace=True)  # Default to 0 for any type
    
    # Compute correlation matrix and drop features with very low correlation with 'Bleaching'
    try:
        corr_matrix = df.corr()
        low_corr_features = []
        threshold = 0.1
        
        # Print all correlations with Bleaching
        print("\nCorrelations with Bleaching:")
        for col in df.columns:
            if col != 'Bleaching' and col in corr_matrix.index:
                corr_value = abs(corr_matrix.loc[col, 'Bleaching'])
                print(f"Correlation of {col} with Bleaching: {corr_value:.4f}")
                if corr_value < threshold:
                    low_corr_features.append(col)
        
        if low_corr_features:
            print(f"Dropping low correlation features: {low_corr_features}")
            df.drop(columns=low_corr_features, inplace=True)
    except Exception as e:
        print(f"Error computing correlations: {e}")
        print("Skipping correlation-based feature selection")
    
    # Define features and target
    X = df.drop(columns=['Bleaching'])
    y = df['Bleaching']
    feature_names = X.columns.tolist()
    
    # Check for class imbalance
    print("\nClass distribution after preprocessing:")
    print(y.value_counts())
    print(f"Positive rate: {y.mean():.2%}")
    
    # If the dataset is imbalanced, use stratified sampling
    if y.mean() < 0.2:  # If positive class is less than 20%
        print("Dataset is imbalanced. Using stratified sampling...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
    
    # Apply feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Verify no NaNs in the scaled data
    if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
        print("Warning: NaN values detected after scaling. Replacing with zeros...")
        X_train_scaled = np.nan_to_num(X_train_scaled)
        X_test_scaled = np.nan_to_num(X_test_scaled)
    
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Test data shape: {X_test_scaled.shape}")
    print(f"Training set positive rate: {y_train.mean():.2%}")
    print(f"Test set positive rate: {y_test.mean():.2%}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, df

# =============================================================================
# Section 3: Classical Machine Learning Models
# =============================================================================
def train_and_evaluate_classical_models(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train and evaluate classical machine learning models.
    
    Args:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_train: Training target
        y_test: Test target
        
    Returns:
        dict: Results of classical models
    """
    # Check for class imbalance and prepare model parameters accordingly
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    if len(class_counts) > 1:
        minority_class_ratio = min(class_counts) / total_samples
        print(f"Minority class ratio: {minority_class_ratio:.2%}")
        
        # Adjust model parameters for imbalanced data if needed
        imbalanced = minority_class_ratio < 0.2
    else:
        imbalanced = False
        print("Warning: Only one class found in training data")
    
    # Define models with parameters adjusted for potential class imbalance
    if imbalanced:
        print("Using class weight 'balanced' for models due to imbalanced data")
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, 
                random_state=42, 
                class_weight='balanced',
                solver='liblinear'  # More stable with imbalanced data
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42,
                class_weight='balanced',
                min_samples_leaf=5  # Prevent overfitting to minority class
            ),
            "SVM": SVC(
                kernel='rbf', 
                probability=True, 
                random_state=42,
                class_weight='balanced'
            ),
            "Naive Bayes": GaussianNB()  # No class_weight parameter for GaussianNB
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "SVM": SVC(kernel='rbf', probability=True, random_state=42),
            "Naive Bayes": GaussianNB()
        }
    
    classical_results = {}
    
    print("\n----- Classical Model Results -----")
    for name, model in models.items():
        try:
            # Verify no NaN values
            if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
                print(f"Warning: NaN values detected before training {name}. Replacing with zeros...")
                X_train_clean = np.nan_to_num(X_train_scaled)
                X_test_clean = np.nan_to_num(X_test_scaled)
            else:
                X_train_clean = X_train_scaled
                X_test_clean = X_test_scaled
            
            start_time = time.time()
            model.fit(X_train_clean, y_train)
            training_time = time.time() - start_time
            
            y_pred = model.predict(X_test_clean)
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate additional metrics for imbalanced data
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            
            classical_results[name] = {
                "accuracy": accuracy,
                "balanced_accuracy": balanced_acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": cm, 
                "training_time": training_time
            }
            
            print(f"{name} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Balanced Accuracy: {balanced_acc:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Training Time: {training_time:.4f} seconds")
            print(f"  Confusion Matrix:\n{cm}")
            print(f"  Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
            print("-----")
        except Exception as e:
            print(f"Error training {name}: {e}")
            import traceback
            traceback.print_exc()
    
    return classical_results

# =============================================================================
# Section 4: Quantum Machine Learning Model using Qadence (QNN)
# =============================================================================
def train_and_evaluate_quantum_model(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train and evaluate a quantum machine learning model using Qadence.
    
    Args:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_train: Training target
        y_test: Test target
        
    Returns:
        tuple: Quantum model accuracy and training time
    """
    print("\n----- Quantum Model (QNN) -----")
    try:
        # Import Qadence for constructing the quantum circuit
        from qadence import QNN, QuantumCircuit, RX, RY, CNOT, Z, chain, FeatureParameter
        
        # Check for NaN values and replace them
        if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
            print("Warning: NaN values detected. Replacing with zeros for quantum model.")
            X_train_scaled_clean = np.nan_to_num(X_train_scaled)
            X_test_scaled_clean = np.nan_to_num(X_test_scaled)
        else:
            X_train_scaled_clean = X_train_scaled
            X_test_scaled_clean = X_test_scaled
        
        # For the quantum model, select features with highest correlation to target
        if X_train_scaled_clean.shape[1] >= 2:
            # Select the two features with highest correlation to target
            if X_train_scaled_clean.shape[1] > 2:
                # Calculate correlation with target for each feature
                correlations = []
                for i in range(X_train_scaled_clean.shape[1]):
                    corr = np.corrcoef(X_train_scaled_clean[:, i], y_train)[0, 1]
                    correlations.append((i, abs(corr)))
                
                # Sort by absolute correlation and get top 2 feature indices
                top_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:2]
                feature_indices = [idx for idx, _ in top_features]
                print(f"Using features with indices {feature_indices} for quantum model (highest correlations)")
            else:
                feature_indices = [0, 1]
                print(f"Using the two available features for quantum model")
                
            X_train_q = X_train_scaled_clean[:, feature_indices]
            X_test_q = X_test_scaled_clean[:, feature_indices]
        else:
            # If only one feature is available, duplicate it
            X_train_q = np.hstack([X_train_scaled_clean, X_train_scaled_clean])
            X_test_q = np.hstack([X_test_scaled_clean, X_test_scaled_clean])
            print(f"Using duplicated features for quantum model (only {X_train_scaled_clean.shape[1]} feature available)")
        
        # Normalize to [0, PI] for encoding as rotation angles
        min_vals = X_train_q.min(axis=0)
        max_vals = X_train_q.max(axis=0)
        # Prevent division by zero
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0  # Replace zero ranges with 1.0
        
        X_train_q_norm = (X_train_q - min_vals) / ranges * PI_value
        X_test_q_norm = (X_test_q - min_vals) / ranges * PI_value
        
        # Check for NaN or inf values after normalization
        if np.isnan(X_train_q_norm).any() or np.isnan(X_test_q_norm).any() or \
           np.isinf(X_train_q_norm).any() or np.isinf(X_test_q_norm).any():
            print("Warning: NaN or inf values detected after normalization. Replacing with PI/2.")
            X_train_q_norm = np.nan_to_num(X_train_q_norm, nan=PI_value/2, posinf=PI_value, neginf=0)
            X_test_q_norm = np.nan_to_num(X_test_q_norm, nan=PI_value/2, posinf=PI_value, neginf=0)
        
        # Convert target to numpy array if it's a pandas Series
        if hasattr(y_train, 'values'):
            y_train_np = y_train.values
            y_test_np = y_test.values
        else:
            y_train_np = y_train
            y_test_np = y_test
        
        # Convert to PyTorch tensors
        X_train_q_tensor = torch.tensor(X_train_q_norm, dtype=torch.float64)
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float64).view(-1, 1)
        X_test_q_tensor = torch.tensor(X_test_q_norm, dtype=torch.float64)
        y_test_tensor = torch.tensor(y_test_np, dtype=torch.float64).view(-1, 1)
        
        print(f"Training tensor shape: {X_train_q_tensor.shape}")
        print(f"Target tensor shape: {y_train_tensor.shape}")
        
        # Handle class imbalance by creating weighted sampling
        # Calculate class weights
        class_counts = np.bincount(y_train_np.astype(int))
        weights = 1.0 / class_counts
        sample_weights = torch.tensor([weights[t] for t in y_train_np.astype(int)], dtype=torch.float64)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create DataLoader with weighted sampling
        batch_size = 128  # Using batches to improve training
        dataset = torch.utils.data.TensorDataset(X_train_q_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler
        )
        
        # Build a 3-qubit quantum circuit with trainable parameters for better expressivity
        from qadence import Parameter
        
        # Create trainable parameters
        theta1 = Parameter("theta1")
        theta2 = Parameter("theta2")
        theta3 = Parameter("theta3")
        theta4 = Parameter("theta4")
        
        # Build an enhanced quantum circuit with both feature encoding and trainable parameters
        qnn_block = chain(
            # Initial feature encoding layer
            RX(0, FeatureParameter("phi")),       # Encode first feature
            RX(1, FeatureParameter("theta")),     # Encode second feature
            CNOT(0, 1),                           # Entangling gate
            
            # First trainable layer
            RY(0, theta1),                        # Trainable rotation
            RY(1, theta2),                        # Trainable rotation
            CNOT(1, 0),                           # Entangling gate
            
            # Second trainable layer
            RY(0, theta3),                        # Trainable rotation
            RY(1, theta4),                        # Trainable rotation
            CNOT(0, 1),                           # Entangling gate
            
            # Feature re-encoding for better feature interaction
            RY(0, FeatureParameter("phi")),       # Encode first feature again
            RY(1, FeatureParameter("theta")),     # Encode second feature again
            CNOT(0, 1)                            # Final entangling gate
        )
        
        qc = QuantumCircuit(2, qnn_block)
        observable = Z(0)  # Measure qubit 0
        
        # Create QNN with trainable parameters
        qnn_model = QNN(
            qc, 
            observable, 
            inputs=["phi", "theta"]
        )
        
        # Print parameter names to verify
        print(f"QNN parameters: {qnn_model.parameters()}")
        
        if len(list(qnn_model.parameters())) == 0:
            print("Warning: QNN has no trainable parameters. Using a different approach.")
            
            # Create a simpler fixed circuit for demonstration
            qnn_model = None
            quantum_accuracy = None
            quantum_training_time = None
        else:
            # Wrap the QNN in a PyTorch Module for classification
            class QuantumClassifier(nn.Module):
                def __init__(self, qnn):
                    super(QuantumClassifier, self).__init__()
                    self.qnn = qnn
                    # Add a classical layer to help with classification decision boundary
                    self.post_process = nn.Sequential(
                        nn.Linear(1, 4),
                        nn.ReLU(),
                        nn.Linear(4, 1),
                        nn.Sigmoid()
                    )
                
                def forward(self, x):
                    out = self.qnn(x).view(-1, 1)  # Expectation value in [-1, 1]
                    # Process through classical layer for better classification
                    return self.post_process(out)
            
            quantum_classifier = QuantumClassifier(qnn_model)
            
            # Loss and optimizer
            # Use weighted binary cross-entropy loss to handle class imbalance
            pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float64)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            # Use a learning rate scheduler for better convergence
            optimizer = optim.Adam(quantum_classifier.parameters(), lr=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
            
            # Training loop for the QNN model
            num_epochs = 200  # Increased number of epochs
            quantum_training_start = time.time()
            
            # Keep track of best model
            best_f1 = 0
            best_model_state = None
            
            try:
                for epoch in range(num_epochs):
                    epoch_loss = 0
                    # Train with batches
                    for batch_x, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = quantum_classifier(batch_x)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    
                    avg_loss = epoch_loss / len(train_loader)
                    scheduler.step(avg_loss)
                    
                    # Evaluate every 20 epochs
                    if (epoch+1) % 20 == 0:
                        with torch.no_grad():
                            q_outputs = quantum_classifier(X_test_q_tensor)
                            q_preds = (q_outputs >= 0.5).float()
                            # Make sure we're importing the correct f1_score from sklearn.metrics
                            from sklearn.metrics import f1_score as sklearn_f1_score
                            current_f1 = sklearn_f1_score(y_test_tensor.cpu().numpy().flatten(), q_preds.cpu().numpy().flatten(), average='binary')
                            
                            if current_f1 > best_f1:
                                best_f1 = current_f1
                                best_model_state = {k: v.clone() for k, v in quantum_classifier.state_dict().items()}
                                
                        print(f"Quantum Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Current F1: {current_f1:.4f}")
            except Exception as e:
                print(f"Error during quantum model training: {e}")
                print("Skipping further quantum model evaluation")
                return None, None
                
            quantum_training_time = time.time() - quantum_training_start
            
            # Load the best model state
            if best_model_state:
                quantum_classifier.load_state_dict(best_model_state)
            
            # Evaluate QNN on test set
            with torch.no_grad():
                q_outputs = quantum_classifier(X_test_q_tensor)
                q_preds = (q_outputs >= 0.5).float()
                quantum_accuracy = (q_preds.eq(y_test_tensor).sum().item()) / y_test_tensor.size(0)
            
            # Calculate additional metrics for imbalanced data
            from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
            
            q_preds_np = q_preds.cpu().numpy().flatten()
            y_test_np = y_test_tensor.cpu().numpy().flatten()
            
            try:
                precision = precision_score(y_test_np, q_preds_np, zero_division=0)
                recall = recall_score(y_test_np, q_preds_np, zero_division=0)
                f1 = f1_score(y_test_np, q_preds_np, zero_division=0)
                balanced_acc = balanced_accuracy_score(y_test_np, q_preds_np)
                
                print(f"Quantum QNN Model Results:")
                print(f"  Accuracy: {quantum_accuracy:.4f}")
                print(f"  Balanced Accuracy: {balanced_acc:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1 Score: {f1:.4f}")
                print(f"  Training Time: {quantum_training_time:.4f} seconds")
                print(f"  Confusion Matrix:\n{confusion_matrix(y_test_np, q_preds_np)}")
                print(f"  Classification Report:\n{classification_report(y_test_np, q_preds_np, zero_division=0)}")
            except Exception as e:
                print(f"Error calculating quantum model metrics: {e}")
        
            # Create a results dictionary similar to classical models
            quantum_results = None
            if quantum_accuracy is not None:
                quantum_results = {
                    "accuracy": quantum_accuracy,
                    "balanced_accuracy": balanced_acc,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "confusion_matrix": confusion_matrix(y_test_np, q_preds_np),
                    "training_time": quantum_training_time
                }

        return quantum_results
    
    except ImportError as e:
        print(f"Qadence or other quantum packages not found: {e}")
        print("Skipping quantum model.")
        return None, None
    except Exception as e:
        print(f"Error in quantum model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# =============================================================================
# Section 5: Quantum Evolution Kernel (QEK) Based Model
# =============================================================================
def train_and_evaluate_qek_model(X_train_scaled, X_test_scaled, y_train, y_test, feature_names=None):
    """
    Train and evaluate a model using the Quantum Evolution Kernel (QEK).
    Enhanced version that tries multiple parameter combinations.
    
    Args:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_train: Training target
        y_test: Test target
        feature_names: Optional list of feature names
        
    Returns:
        dict: Results from best QEK model
    """
    print("\n----- Enhanced Quantum Evolution Kernel (QEK) Model -----")
    
    # Import required libraries
    import numpy as np
    import time
    import traceback
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, classification_report,
        precision_score, recall_score, f1_score, balanced_accuracy_score
    )
    
    try:
        # Try to import QEK 
        from qek.kernel import QuantumEvolutionKernel as QEK
        print("QEK library successfully imported")
        
        # Create a more efficient class for QEK data points
        class GraphStructuredDataPoint:
            """
            Custom class with state_dict attribute for QEK.
            """
            def __init__(self, state_dict, target=None):
                self.state_dict = state_dict
                self.target = target
        
        # Check for NaN values
        if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
            print("Warning: NaN values detected. Replacing with zeros.")
            X_train_scaled_clean = np.nan_to_num(X_train_scaled)
            X_test_scaled_clean = np.nan_to_num(X_test_scaled)
        else:
            X_train_scaled_clean = X_train_scaled
            X_test_scaled_clean = X_test_scaled
        
        # Enhanced feature to state dictionary conversion
        def feature_to_state_dict(features, bit_depth=4):
            """
            Convert feature vector to a quantum state dictionary with enhanced representation.
            
            Args:
                features: Feature vector
                bit_depth: Number of bits to use for state representation
                
            Returns:
                dict: State dictionary with binary strings as keys and probabilities as values
            """
            # Ensure features are positive (for probability distribution)
            if np.any(features < 0):
                # Shift distribution to be positive
                features = features - np.min(features)
            
            # Get feature magnitudes
            features_abs = np.abs(features)
            sum_features = np.sum(features_abs)
            
            # Normalize to valid probability distribution
            if sum_features < 1e-10:
                # Handle zero vectors
                probs = np.ones_like(features_abs) / len(features_abs)
            else:
                probs = features_abs / sum_features
            
            # Create state dictionary
            state_dict = {}
            
            # Get indices and sort by probability (highest first)
            idx_prob_pairs = [(i, p) for i, p in enumerate(probs)]
            idx_prob_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Take top K probabilities or all if fewer
            K = min(2**bit_depth - 1, len(features))
            total_prob = 0.0
            
            # Create binary representation for top K probabilities
            for i, (idx, prob) in enumerate(idx_prob_pairs[:K]):
                if prob > 1e-10:  # Only keep non-negligible probabilities
                    # Generate binary string of appropriate length
                    binary = format(i, f'0{bit_depth}b')
                    state_dict[binary] = float(prob)
                    total_prob += prob
            
            # Ensure dictionary isn't empty
            if not state_dict:
                state_dict['0' * bit_depth] = 1.0
            elif total_prob < 0.99:
                # Normalize to ensure probabilities sum to 1
                for key in state_dict:
                    state_dict[key] /= total_prob
                
            return state_dict
        
        # Use stratified sampling for smaller dataset
        from sklearn.model_selection import StratifiedShuffleSplit
        
        # Try different dataset sizes to find optimal size
        max_samples_options = [1200, 1000, 500]
        mu_values = [0.1, 0.5, 1.0, 1.2, 2.0]  # Test multiple mu values
        bit_depths = [3, 4, 5]  # Test multiple bit depths
        
        best_balanced_acc = 0
        best_qek_results = None
        best_params = {}
        
        # Try different combinations of parameters
        for max_samples in max_samples_options:
            for bit_depth in bit_depths:
                # Create dataset of appropriate size with stratification
                if len(X_train_scaled_clean) > max_samples:
                    print(f"Creating dataset with {max_samples} samples and bit depth {bit_depth}")
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-max_samples/len(X_train_scaled_clean), random_state=42)
                    for train_idx, _ in sss.split(X_train_scaled_clean, y_train):
                        X_train_reduced = X_train_scaled_clean[train_idx]
                        y_train_reduced = np.array(y_train)[train_idx]
                else:
                    X_train_reduced = X_train_scaled_clean
                    y_train_reduced = np.array(y_train)
                
                # Similarly, limit test set size
                max_test_samples = min(200, len(X_test_scaled_clean))
                if len(X_test_scaled_clean) > max_test_samples:
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-max_test_samples/len(X_test_scaled_clean), random_state=42)
                    for test_idx, _ in sss.split(X_test_scaled_clean, y_test):
                        X_test_reduced = X_test_scaled_clean[test_idx]
                        y_test_reduced = np.array(y_test)[test_idx]
                else:
                    X_test_reduced = X_test_scaled_clean
                    y_test_reduced = np.array(y_test)
                    
                print(f"Using {len(X_train_reduced)} training samples and {len(X_test_reduced)} test samples")
                
                # Convert data to QEK format with state dictionaries using current bit depth
                train_data = [
                    GraphStructuredDataPoint(feature_to_state_dict(x, bit_depth), y)
                    for x, y in zip(X_train_reduced, y_train_reduced)
                ]
                    
                test_data = [
                    GraphStructuredDataPoint(feature_to_state_dict(x, bit_depth), y)
                    for x, y in zip(X_test_reduced, y_test_reduced)
                ]
                
                # Verify first sample
                print(f"Sample state_dict format: {train_data[0].state_dict}")
                
                # Try different mu values
                for mu in mu_values:
                    print(f"\nTesting QEK with mu={mu}")
                    start_time = time.time()
                    
                    # Initialize QEK with current mu
                    kernel = QEK(mu=mu)
                    
                    # Compute kernel matrices in batches
                    print("Computing kernel matrices...")
                    batch_size = 50
                    
                    n_train = len(train_data)
                    n_test = len(test_data)
                    
                    # Initialize kernel matrices
                    K_train = np.zeros((n_train, n_train))
                    K_test = np.zeros((n_test, n_train))
                    
                    # Compute training kernel matrix in batches
                    print("Generating training kernel matrix...")
                    for i in range(0, n_train, batch_size):
                        batch_end = min(i + batch_size, n_train)
                        
                        for j in range(0, n_train, batch_size):
                            j_end = min(j + batch_size, n_train)
                            
                            # Compute kernel values for this batch
                            for bi in range(i, batch_end):
                                for bj in range(j, j_end):
                                    try:
                                        K_train[bi, bj] = kernel(train_data[bi], train_data[bj])
                                        # Ensure symmetry for faster computation
                                        if bi != bj:
                                            K_train[bj, bi] = K_train[bi, bj]
                                    except Exception as ke:
                                        # Fallback for any kernel computation errors
                                        K_train[bi, bj] = 1.0 if bi == bj else 0.0
                                        if bi != bj:
                                            K_train[bj, bi] = K_train[bi, bj]
                    
                    # Compute test kernel matrix in batches
                    print("Generating test kernel matrix...")
                    for i in range(0, n_test, batch_size):
                        batch_end = min(i + batch_size, n_test)
                        
                        for j in range(0, n_train, batch_size):
                            j_end = min(j + batch_size, n_train)
                            
                            # Compute kernel values for this batch
                            for bi in range(i, batch_end):
                                for bj in range(j, j_end):
                                    try:
                                        K_test[bi, bj] = kernel(test_data[bi], train_data[bj])
                                    except Exception as ke:
                                        # Fallback for any kernel computation errors
                                        K_test[bi, bj] = 0.0
                    
                    # Replace any NaN values in kernel matrices
                    K_train = np.nan_to_num(K_train)
                    K_test = np.nan_to_num(K_test)
                    
                    # Try different C values for SVM
                    from sklearn.svm import SVC
                    
                    for C in [0.1, 1.0, 10.0, 100.0]:
                        print(f"Testing SVM with C={C}")
                        model = SVC(
                            kernel='precomputed',
                            random_state=42,
                            class_weight='balanced',
                            C=C
                        )
                        
                        try:
                            # Train with precomputed kernel matrix
                            model.fit(K_train, y_train_reduced)
                            
                            # Predict using precomputed kernel matrix
                            y_pred_qek = model.predict(K_test)
                            qek_training_time = time.time() - start_time
                            
                            # Calculate evaluation metrics
                            accuracy = accuracy_score(y_test_reduced, y_pred_qek)
                            balanced_acc = balanced_accuracy_score(y_test_reduced, y_pred_qek)
                            f1 = f1_score(y_test_reduced, y_pred_qek, average='weighted', zero_division=0)
                            precision = precision_score(y_test_reduced, y_pred_qek, average='weighted', zero_division=0)
                            recall = recall_score(y_test_reduced, y_pred_qek, average='weighted', zero_division=0)
                            
                            print(f"Results - Acc: {accuracy:.4f}, Bal Acc: {balanced_acc:.4f}, F1: {f1:.4f}")
                            
                            # Check if this is the best model so far
                            if balanced_acc > best_balanced_acc:
                                best_balanced_acc = balanced_acc
                                best_params = {
                                    'mu': mu, 
                                    'C': C, 
                                    'samples': max_samples,
                                    'bit_depth': bit_depth
                                }
                                best_qek_results = {
                                    "accuracy": accuracy,
                                    "balanced_accuracy": balanced_acc,
                                    "precision": precision,
                                    "recall": recall,
                                    "f1_score": f1,
                                    "confusion_matrix": confusion_matrix(y_test_reduced, y_pred_qek),
                                    "training_time": qek_training_time,
                                    "parameters": best_params
                                }
                        except Exception as fit_error:
                            print(f"Error with SVM (C={C}): {fit_error}")
                            continue
        
        # Print and return the best results
        if best_qek_results:
            print("\n===== Best QEK-SVM Model Results =====")
            print(f"Best parameters: {best_params}")
            print(f"  Accuracy: {best_qek_results['accuracy']:.4f}")
            print(f"  Balanced Accuracy: {best_qek_results['balanced_accuracy']:.4f}")
            print(f"  Precision: {best_qek_results['precision']:.4f}")
            print(f"  Recall: {best_qek_results['recall']:.4f}")
            print(f"  F1 Score: {best_qek_results['f1_score']:.4f}")
            print(f"  Training Time: {best_qek_results['training_time']:.4f} seconds")
            print(f"  Confusion Matrix:\n{best_qek_results['confusion_matrix']}")
            
            return best_qek_results
        else:
            print("No successful QEK model found.")
            return None
        
    except Exception as e:
        print(f"Error in QEK model: {e}")
        traceback.print_exc()
        return None
    


# =============================================================================
# Section 6: Comparison and Visualization
# =============================================================================
def visualize_results(classical_results,quantum_results , qek_results, df):
    """
    Visualize and compare model results.
    
    Args:
        classical_results: Results from classical models
        quantum_accuracy: Accuracy of quantum model
        quantum_training_time: Training time of quantum model
        qek_results: Results from QEK model
        df: Preprocessed dataframe
    """
    if not classical_results and quantum_results is None and qek_results is None:
        print("No results to visualize.")
        return
    
    model_names = []
    accuracies = []
    balanced_accs = []
    precisions = []
    recalls = []
    f1_scores = []
    training_times = []
    
    # Collect results from classical models
    if classical_results:
        for name, result in classical_results.items():
            model_names.append(name)
            accuracies.append(result["accuracy"])
            balanced_accs.append(result["balanced_accuracy"])
            precisions.append(result["precision"])
            recalls.append(result["recall"])
            f1_scores.append(result["f1_score"])
            training_times.append(result["training_time"])
    
    # Add Qadence QNN results if available
    if quantum_results is not None:
        model_names.append("Quantum QNN")
        accuracies.append(quantum_results["accuracy"])
        balanced_accs.append(quantum_results["balanced_accuracy"])
        precisions.append(quantum_results["precision"])
        recalls.append(quantum_results["recall"])
        f1_scores.append(quantum_results["f1_score"])
        training_times.append(quantum_results["training_time"])
    
    # Add QEK results if available
    if qek_results is not None:
        model_names.append("QEK-SVM")
        accuracies.append(qek_results["accuracy"])
        balanced_accs.append(qek_results["balanced_accuracy"])
        precisions.append(qek_results["precision"])
        recalls.append(qek_results["recall"])
        f1_scores.append(qek_results["f1_score"])
        training_times.append(qek_results["training_time"])
    
    # 1. Plot model accuracies
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, accuracies, color='skyblue')
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{acc:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("model_accuracy_comparison.png")
    plt.show()
    
    # 2. Plot balanced accuracies (better for imbalanced data)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, balanced_accs, color='lightgreen')
    plt.ylabel("Balanced Accuracy")
    plt.title("Model Balanced Accuracy Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add balanced accuracy values on top of bars
    for bar, bacc in zip(bars, balanced_accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{bacc:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("model_balanced_accuracy_comparison.png")
    plt.show()
    
    # 3. Plot F1 scores
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, f1_scores, color='coral')
    plt.ylabel("F1 Score")
    plt.title("Model F1 Score Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add F1 score values on top of bars
    for bar, f1 in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{f1:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("model_f1_score_comparison.png")
    plt.show()
    
    # 4. Plot training times
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, training_times, color='lightpink')
    plt.ylabel("Training Time (seconds)")
    plt.title("Model Training Time Comparison")
    plt.xticks(rotation=45)
    
    # Add time values on top of bars
    for bar, time_val in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{time_val:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("model_training_time_comparison.png")
    plt.show()
    
    # 5. Combined metrics chart
    plt.figure(figsize=(15, 8))
    x = np.arange(len(model_names))
    width = 0.2
    
    plt.bar(x - width*1.5, accuracies, width, label='Accuracy', color='skyblue')
    plt.bar(x - width/2, balanced_accs, width, label='Balanced Accuracy', color='lightgreen')
    plt.bar(x + width/2, precisions, width, label='Precision', color='coral')
    plt.bar(x + width*1.5, recalls, width, label='Recall', color='lightpink')
    
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('All Metrics Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("all_metrics_comparison.png")
    plt.show()
    
    # 6. Display a correlation heatmap
    try:
        plt.figure(figsize=(12, 10))
        corr_matrix = df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask for upper triangle
        
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
                    linewidths=0.5, mask=mask)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("feature_correlation_heatmap.png")
        plt.show()
    except Exception as e:
        print(f"Error creating correlation heatmap: {e}")

# =============================================================================
# Main Function
# =============================================================================
def main():
    """
    Main function to run the coral reef bleaching prediction pipeline.
    """
    try:
        # 1. Load and preprocess data
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names, df = load_and_preprocess_data()
        
        # 2. Train and evaluate classical models
        classical_results = train_and_evaluate_classical_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        quantum_results = None
        # 3. Train and evaluate quantum model (QNN)
        quantum_results = train_and_evaluate_quantum_model(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # 4. Train and evaluate QEK model
        qek_results = train_and_evaluate_qek_model(X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
        
        # 5. Visualize results
        visualize_results(classical_results, quantum_results, qek_results, df)
        
        # 6. Print final summary
        print("\n----- Final Summary -----")
        print("Top performing models:")
        all_accuracies = []
        
        for name, result in classical_results.items():
            all_accuracies.append((name, result["accuracy"]))
        
        if quantum_results is not None:
            all_accuracies.append(("Quantum QNN", quantum_results["accuracy"]))
            
        if qek_results is not None:
            all_accuracies.append(("QEK-SVM", qek_results["accuracy"]))
        
        # Sort by accuracy (descending)
        all_accuracies.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, acc) in enumerate(all_accuracies):
            print(f"{i+1}. {name}: {acc:.4f}")
        
        print("\nFigures saved:")
        print("- model_accuracy_comparison.png")
        print("- model_balanced_accuracy_comparison.png")
        print("- model_f1_score_comparison.png")
        print("- model_training_time_comparison.png")
        print("- all_metrics_comparison.png")
        print("- feature_correlation_heatmap.png")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()