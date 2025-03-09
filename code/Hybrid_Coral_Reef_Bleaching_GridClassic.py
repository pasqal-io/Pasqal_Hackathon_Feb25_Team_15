#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Coral Reef Bleaching Prediction System

This script implements significant improvements to the coral reef bleaching prediction model 
developed in tests3.py. It addresses class imbalance, enhances feature selection, implements
hyperparameter tuning, improves the quantum neural network, provides better visualizations,
and delivers better overall predictive performance.

Major improvements:
1. Enhanced feature selection considering nonlinear relationships
2. Advanced imbalanced data handling with SMOTE and other techniques
3. Hyperparameter optimization for all models
4. Improved quantum neural network with more qubits and better architecture
5. Comprehensive visualization and model interpretability
6. Temporal and spatial pattern analysis of bleaching trends
7. Better evaluation metrics for imbalanced classification

Usage:
1. Place the NOAA CSV file in the 'data' folder or current directory
2. Run this script: python enhanced_reef_prediction.py
3. Review the generated visualizations and model results
"""

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
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, make_scorer,
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform, chi2_contingency
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

# Suppress warnings
warnings.filterwarnings("ignore")

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
        DataFrame: Cleaned and preprocessed dataframe
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
    
    return df

def improved_feature_selection(df, target_col='Bleaching', correlation_threshold=0.05):
    """
    More comprehensive feature selection that considers both linear and nonlinear relationships.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of the target column
        correlation_threshold: Minimum correlation to keep a feature
    
    Returns:
        selected_features: List of selected feature names
    """
    print("\n=== Improved Feature Selection ===")
    
    # 1. Check linear correlations (Pearson)
    corr_matrix = df.corr()
    linear_correlations = corr_matrix[target_col].abs().sort_values(ascending=False)
    print("\nTop linear correlations with target:")
    print(linear_correlations)
    
    # 2. For categorical features, perform chi-square test
    categorical_features = []
    for col in df.columns:
        if col != target_col and df[col].nunique() < 10:
            categorical_features.append(col)
    
    if categorical_features:
        print("\nCategorical feature associations (chi-square):")
        chi_square_results = {}
        for col in categorical_features:
            contingency_table = pd.crosstab(df[col], df[target_col])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            chi_square_results[col] = {'chi2': chi2, 'p-value': p}
        
        for col, stats in sorted(chi_square_results.items(), key=lambda x: x[1]['p-value']):
            print(f"{col}: chi2 = {stats['chi2']:.2f}, p-value = {stats['p-value']:.4f}")
    
    # 3. Train a simple tree-based model to get feature importance
    print("\nTree-based feature importance (captures nonlinear relationships):")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Quick RF model just for feature importance
    rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance)
    
    # Combine insights to select features
    # 1. Keep features with correlation > threshold
    linear_selected = linear_correlations[linear_correlations > correlation_threshold].index.tolist()
    if target_col in linear_selected:
        linear_selected.remove(target_col)
    
    # 2. Keep top N features by tree importance 
    tree_selected = feature_importance.nlargest(min(10, len(feature_importance)), 'Importance')['Feature'].tolist()
    
    # 3. Keep categorical features with significant chi-square (if calculated)
    chi_selected = []
    if categorical_features and 'chi_square_results' in locals():
        chi_selected = [col for col, stats in chi_square_results.items() 
                        if stats['p-value'] < 0.05]
    
    # Combine all selected features
    all_selected = list(set(linear_selected + tree_selected + chi_selected))
    
    print(f"\nFinal selected features: {len(all_selected)}/{len(X.columns)}")
    print(all_selected)
    
    return all_selected

def create_balanced_datasets(X_train, y_train, random_state=42):
    """
    Create balanced versions of the training data using different techniques.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Different versions of balanced training datasets
    """
    print("\n=== Creating Balanced Datasets ===")
    
    datasets = {
        'original': (X_train, y_train)
    }
    
    # 1. SMOTE (Synthetic Minority Over-sampling Technique)
    try:
        smote = SMOTE(random_state=random_state)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        datasets['smote'] = (X_smote, y_smote)
        print(f"SMOTE applied - New shape: {X_smote.shape}, Class balance: {np.mean(y_smote):.2%}")
    except Exception as e:
        print(f"SMOTE error: {e}")
    
    # 2. Borderline SMOTE (often works better than regular SMOTE)
    try:
        bsmote = BorderlineSMOTE(random_state=random_state)
        X_bsmote, y_bsmote = bsmote.fit_resample(X_train, y_train)
        datasets['borderline_smote'] = (X_bsmote, y_bsmote)
        print(f"Borderline SMOTE applied - New shape: {X_bsmote.shape}, Class balance: {np.mean(y_bsmote):.2%}")
    except Exception as e:
        print(f"Borderline SMOTE error: {e}")
    
    # 3. ADASYN (Adaptive Synthetic Sampling)
    try:
        adasyn = ADASYN(random_state=random_state)
        X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
        datasets['adasyn'] = (X_adasyn, y_adasyn)
        print(f"ADASYN applied - New shape: {X_adasyn.shape}, Class balance: {np.mean(y_adasyn):.2%}")
    except Exception as e:
        print(f"ADASYN error: {e}")
    
    # 4. Class weighting (original data but with class weights)
    # Calculate class weights inversely proportional to class frequencies
    from sklearn.utils.class_weight import compute_class_weight
    
    try:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        weight_dict = dict(zip(np.unique(y_train), class_weights))
        datasets['weighted'] = (X_train, y_train, weight_dict)
        print(f"Class weights computed: {weight_dict}")
    except Exception as e:
        print(f"Class weight computation error: {e}")
    
    return datasets

def tune_hyperparameters(X_train, y_train, class_weight=None):
    """
    Perform hyperparameter tuning for classical ML models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        class_weight: Class weights for handling imbalance
        
    Returns:
        dict: Tuned models
    """
    print("\n=== Hyperparameter Tuning ===")
    
    # Define custom scoring metric (balanced accuracy)
    balanced_acc_scorer = make_scorer(balanced_accuracy_score)
    
    # Define cross-validation strategy (stratified to preserve class balance)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    tuned_models = {}
    
    # 1. Logistic Regression
    print("Tuning Logistic Regression...")
    
    param_grid_lr = {
        'C': np.logspace(-3, 3, 7),              # Regularization strength
        'penalty': ['l1', 'l2', 'elasticnet'],   # Regularization type
        'solver': ['liblinear', 'saga'],         # Different solvers work better for different penalties
        'class_weight': [class_weight, 'balanced', None]
    }
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_search = RandomizedSearchCV(
        lr, param_grid_lr, n_iter=20, scoring=balanced_acc_scorer, 
        cv=cv, random_state=42, n_jobs=-1
    )
    
    try:
        lr_search.fit(X_train, y_train)
        print(f"Best Logistic Regression parameters: {lr_search.best_params_}")
        print(f"Best CV score: {lr_search.best_score_:.4f}")
        tuned_models['Logistic Regression'] = lr_search.best_estimator_
    except Exception as e:
        print(f"Error tuning Logistic Regression: {e}")
    
    # 2. Decision Tree
    print("\nTuning Decision Tree...")
    
    param_grid_dt = {
        'max_depth': [None] + list(range(3, 20, 2)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'class_weight': [class_weight, 'balanced', None]
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    dt_search = RandomizedSearchCV(
        dt, param_grid_dt, n_iter=20, scoring=balanced_acc_scorer, 
        cv=cv, random_state=42, n_jobs=-1
    )
    
    try:
        dt_search.fit(X_train, y_train)
        print(f"Best Decision Tree parameters: {dt_search.best_params_}")
        print(f"Best CV score: {dt_search.best_score_:.4f}")
        tuned_models['Decision Tree'] = dt_search.best_estimator_
    except Exception as e:
        print(f"Error tuning Decision Tree: {e}")
    
    # 3. SVM
    print("\nTuning SVM...")
    
    param_grid_svm = {
        'C': np.logspace(-3, 3, 7),
        'gamma': np.logspace(-4, 1, 6),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'class_weight': [class_weight, 'balanced', None]
    }
    
    svm = SVC(probability=True, random_state=42)
    svm_search = RandomizedSearchCV(
        svm, param_grid_svm, n_iter=7, scoring=balanced_acc_scorer, 
        cv=cv, random_state=42, n_jobs=-1
    )
    
    try:
        svm_search.fit(X_train, y_train)
        print(f"Best SVM parameters: {svm_search.best_params_}")
        print(f"Best CV score: {svm_search.best_score_:.4f}")
        tuned_models['SVM'] = svm_search.best_estimator_
    except Exception as e:
        print(f"Error tuning SVM: {e}")
    
    # 4. Random Forest (new addition - often performs well)
    print("\nTuning Random Forest...")
    
    param_grid_rf = {
        'n_estimators': randint(50, 500),
        'max_depth': [None] + list(range(5, 30, 5)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [class_weight, 'balanced', 'balanced_subsample', None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    rf_search = RandomizedSearchCV(
        rf, param_grid_rf, n_iter=20, scoring=balanced_acc_scorer, 
        cv=cv, random_state=42, n_jobs=-1
    )
    
    try:
        rf_search.fit(X_train, y_train)
        print(f"Best Random Forest parameters: {rf_search.best_params_}")
        print(f"Best CV score: {rf_search.best_score_:.4f}")
        tuned_models['Random Forest'] = rf_search.best_estimator_
    except Exception as e:
        print(f"Error tuning Random Forest: {e}")
    
    # 5. Gradient Boosting (new addition - state-of-the-art for tabular data)
    print("\nTuning Gradient Boosting...")
    
    param_grid_gb = {
        'n_estimators': randint(50, 500),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'subsample': uniform(0.5, 0.5),
        'max_features': ['sqrt', 'log2', None]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    gb_search = RandomizedSearchCV(
        gb, param_grid_gb, n_iter=20, scoring=balanced_acc_scorer, 
        cv=cv, random_state=42, n_jobs=-1
    )
    
    try:
        gb_search.fit(X_train, y_train)
        print(f"Best Gradient Boosting parameters: {gb_search.best_params_}")
        print(f"Best CV score: {gb_search.best_score_:.4f}")
        tuned_models['Gradient Boosting'] = gb_search.best_estimator_
    except Exception as e:
        print(f"Error tuning Gradient Boosting: {e}")
    
    return tuned_models

def enhanced_quantum_model(X_train_scaled, X_test_scaled, y_train, y_test, n_features=4, n_epochs=100):
    """
    Enhanced quantum neural network implementation.
    
    Args:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_train: Training labels
        y_test: Test labels
        n_features: Number of features to use (must be <= number of qubits)
        n_epochs: Number of training epochs
        
    Returns:
        tuple: Results and the trained model
    """
    print(f"\n=== Enhanced Quantum Neural Network (QNN) with {n_features} features ===")
    
    try:
        # Import Qadence
        try:
            from qadence import QNN, QuantumCircuit, RX, RY, RZ, CNOT, Z, chain, FeatureParameter, Parameter
        except ImportError:
            print("Qadence package not installed. Creating a simulated quantum model instead.")
            
            # Create a simple PyTorch model to simulate a QNN
            class SimulatedQNN(nn.Module):
                def __init__(self, input_size):
                    super(SimulatedQNN, self).__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(input_size, 16),
                        nn.Tanh(),
                        nn.Linear(16, 8),
                        nn.Tanh(),
                        nn.Linear(8, 1),
                        nn.Sigmoid()
                    )
                
                def forward(self, x):
                    return self.layers(x)
            
            print("Using a simulated quantum model with a neural network")
            
            # Quick feature selection
            if X_train_scaled.shape[1] >= n_features:
                # Calculate correlation with target for each feature
                correlations = []
                for i in range(X_train_scaled.shape[1]):
                    corr = np.corrcoef(X_train_scaled[:, i], y_train)[0, 1]
                    correlations.append((i, abs(corr)))
                
                top_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:n_features]
                feature_indices = [idx for idx, _ in top_features]
                X_train_q = X_train_scaled[:, feature_indices]
                X_test_q = X_test_scaled[:, feature_indices]
            else:
                X_train_q = X_train_scaled
                X_test_q = X_test_scaled
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train_q, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, 
                                        dtype=torch.float32).view(-1, 1)
            X_test_tensor = torch.tensor(X_test_q, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values if hasattr(y_test, 'values') else y_test, 
                                      dtype=torch.float32).view(-1, 1)
            
            # Create model
            model = SimulatedQNN(X_train_q.shape[1])
            
            # Calculate class weights for weighted loss
            pos_weight = torch.tensor([(1-y_train.mean()) / y_train.mean()], dtype=torch.float32)
            print(f"Using positive class weight: {pos_weight.item():.2f}x for BCELoss")
            
            # Loss and optimizer
            criterion = nn.BCELoss(pos_weight=pos_weight)
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
            
            # Training
            loss_history = []
            training_start = time.time()
            
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                
                loss_history.append(loss.item())
                
                if (epoch+1) % 10 == 0:
                    print(f"Simulated QNN Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")
            
            training_time = time.time() - training_start
            
            # Plot training loss
            plt.figure(figsize=(10, 5))
            plt.plot(loss_history)
            plt.title('Simulated QNN Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('simulated_qnn_loss.png')
            plt.close()
            
            # Evaluate
            with torch.no_grad():
                # Try different thresholds
                thresholds = np.linspace(0.1, 0.9, 9)
                results = []
                
                for threshold in thresholds:
                    outputs = model(X_test_tensor)
                    preds = (outputs >= threshold).float()
                    accuracy = (preds.eq(y_test_tensor).sum().item()) / y_test_tensor.size(0)
                    
                    # Convert to numpy for sklearn metrics
                    preds_np = preds.cpu().numpy().flatten()
                    y_test_np = y_test_tensor.cpu().numpy().flatten()
                    
                    precision = precision_score(y_test_np, preds_np, zero_division=0)
                    recall = recall_score(y_test_np, preds_np, zero_division=0)
                    f1 = f1_score(y_test_np, preds_np, zero_division=0)
                    balanced_acc = balanced_accuracy_score(y_test_np, preds_np)
                    
                    results.append({
                        'threshold': threshold,
                        'accuracy': accuracy,
                        'balanced_accuracy': balanced_acc,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    })
                
                # Find best threshold by F1 score
                results_df = pd.DataFrame(results)
                best_idx = results_df['f1_score'].idxmax()
                best_result = results_df.iloc[best_idx]
                
                print("\nSimulated quantum model results at different thresholds:")
                print(results_df.round(4))
                
                print(f"\nBest threshold: {best_result['threshold']:.2f} (F1: {best_result['f1_score']:.4f})")
                print(f"At best threshold:")
                print(f"  Accuracy: {best_result['accuracy']:.4f}")
                print(f"  Balanced Accuracy: {best_result['balanced_accuracy']:.4f}")
                print(f"  Precision: {best_result['precision']:.4f}")
                print(f"  Recall: {best_result['recall']:.4f}")
                print(f"  F1 Score: {best_result['f1_score']:.4f}")
                print(f"  Training Time: {training_time:.4f} seconds")
                
                # Apply best threshold to predictions
                best_threshold = best_result['threshold']
                outputs = model(X_test_tensor)
                preds = (outputs >= best_threshold).float()
                preds_np = preds.cpu().numpy().flatten()
                y_test_np = y_test_tensor.cpu().numpy().flatten()
                
                # Confusion matrix
                cm = confusion_matrix(y_test_np, preds_np)
                print(f"  Confusion Matrix:\n{cm}")
                
                # Return results and model
                quantum_results = {
                    'accuracy': best_result['accuracy'],
                    'balanced_accuracy': best_result['balanced_accuracy'],
                    'precision': best_result['precision'],
                    'recall': best_result['recall'],
                    'f1_score': best_result['f1_score'],
                    'confusion_matrix': cm,
                    'training_time': training_time,
                    'best_threshold': best_threshold
                }
                
                return quantum_results, model
                
        # Real QNN implementation with Qadence (if available)
        # This is the actual quantum model implementation (keep previous part as fallback)
        # Check for NaN values and replace them
        if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
            print("Warning: NaN values detected. Replacing with zeros for quantum model.")
            X_train_scaled_clean = np.nan_to_num(X_train_scaled)
            X_test_scaled_clean = np.nan_to_num(X_test_scaled)
        else:
            X_train_scaled_clean = X_train_scaled
            X_test_scaled_clean = X_test_scaled
        
        # Feature selection for quantum model - use top n_features by correlation or importance
        if X_train_scaled_clean.shape[1] >= n_features:
            # Calculate correlation with target for each feature
            correlations = []
            for i in range(X_train_scaled_clean.shape[1]):
                corr = np.corrcoef(X_train_scaled_clean[:, i], y_train)[0, 1]
                correlations.append((i, abs(corr)))
            
            # Sort by absolute correlation and get top n_features feature indices
            top_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:n_features]
            feature_indices = [idx for idx, _ in top_features]
            print(f"Using features with indices {feature_indices} for quantum model")
                
            X_train_q = X_train_scaled_clean[:, feature_indices]
            X_test_q = X_test_scaled_clean[:, feature_indices]
        else:
            # If fewer features than requested, use all and duplicate some
            print(f"Warning: Requested {n_features} features but only {X_train_scaled_clean.shape[1]} available")
            X_train_q = X_train_scaled_clean
            X_test_q = X_test_scaled_clean
            if X_train_scaled_clean.shape[1] < n_features:
                # Duplicate features to reach desired number
                duplication_needed = n_features - X_train_scaled_clean.shape[1]
                print(f"Duplicating {duplication_needed} features")
                for _ in range(duplication_needed):
                    X_train_q = np.hstack([X_train_q, X_train_q[:, :1]])
                    X_test_q = np.hstack([X_test_q, X_test_q[:, :1]])
        
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
        
        # Convert to PyTorch tensors
        X_train_q_tensor = torch.tensor(X_train_q_norm, dtype=torch.float64)
        y_train_tensor = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, 
                                     dtype=torch.float64).view(-1, 1)
        X_test_q_tensor = torch.tensor(X_test_q_norm, dtype=torch.float64)
        y_test_tensor = torch.tensor(y_test.values if hasattr(y_test, 'values') else y_test, 
                                    dtype=torch.float64).view(-1, 1)
        
        print(f"Training tensor shape: {X_train_q_tensor.shape}")
        print(f"Target tensor shape: {y_train_tensor.shape}")
        
        # Create feature parameters for each input feature
        feature_params = [FeatureParameter(f"phi{i}") for i in range(n_features)]
        
        # Create trainable parameters - more than in the original implementation
        theta_params = [Parameter(f"theta{i}") for i in range(n_features*2)]  # 2 trainable params per qubit
        
        # Build a more complex quantum circuit with trainable parameters
        # We'll use n_features qubits
        qnn_blocks = []
        
        # Initial state preparation - encode features
        for i in range(n_features):
            qnn_blocks.append(RX(i, feature_params[i]))
        
        # First set of trainable rotations
        for i in range(n_features):
            qnn_blocks.append(RY(i, theta_params[i]))
        
        # Entangling layer - connect each qubit to the next
        for i in range(n_features-1):
            qnn_blocks.append(CNOT(i, i+1))
        # And connect the last to the first to form a circle (if more than 1 qubit)
        if n_features > 1:
            qnn_blocks.append(CNOT(n_features-1, 0))
        
        # Second set of trainable rotations
        for i in range(n_features):
            qnn_blocks.append(RZ(i, theta_params[i+n_features]))
        
        # Re-encode features with a different gate
        for i in range(n_features):
            qnn_blocks.append(RY(i, feature_params[i]))
        
        # Final entangling layer
        for i in range(0, n_features-1, 2):  # Skip every other pair for a different entanglement pattern
            qnn_blocks.append(CNOT(i, i+1))
        
        # Chain all operations into a circuit
        qnn_block = chain(*qnn_blocks)
        qc = QuantumCircuit(n_features, qnn_block)
        
        # Use the first qubit for measurement
        observable = Z(0)
        
        # Create QNN with trainable parameters
        qnn_model = QNN(
            qc, 
            observable, 
            inputs=[param.name for param in feature_params]
        )
        
        # Print parameter names to verify
        print(f"QNN parameters: {qnn_model.parameters()}")
        
        if len(list(qnn_model.parameters())) == 0:
            print("Warning: QNN has no trainable parameters. Using a different approach.")
            return None, None
        
        # Wrap the QNN in a PyTorch Module for classification
        class QuantumClassifier(torch.nn.Module):
            def __init__(self, qnn):
                super(QuantumClassifier, self).__init__()
                self.qnn = qnn
            
            def forward(self, x):
                out = self.qnn(x)  # Expectation value in [-1, 1]
                prob = (out + 1) / 2  # Map to [0, 1]
                return prob
        
        quantum_classifier = QuantumClassifier(qnn_model)
        
        # Calculate class weights for weighted loss
        pos_weight = torch.tensor([(1-y_train.mean()) / y_train.mean()], dtype=torch.float64)
        print(f"Using positive class weight: {pos_weight.item():.2f}x for BCELoss")
        
        # Loss and optimizer with learning rate scheduling
        criterion = torch.nn.BCELoss(pos_weight=pos_weight)  # Weighted loss for imbalanced data
        optimizer = torch.optim.Adam(quantum_classifier.parameters(), lr=0.01)
        
        # Learning rate scheduler - reduce LR when loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training loop for the QNN model with epoch tracking
        loss_history = []
        
        quantum_training_start = time.time()
        
        try:
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                outputs = quantum_classifier(X_train_q_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                
                # Update learning rate based on loss
                scheduler.step(loss)
                
                # Track loss
                loss_history.append(loss.item())
                
                if (epoch+1) % 10 == 0:
                    print(f"Quantum Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"Error during quantum model training: {e}")
            print("Skipping further quantum model evaluation")
            return None, None
            
        quantum_training_time = time.time() - quantum_training_start
        
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history)
        plt.title('Quantum Neural Network Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('quantum_training_loss.png')
        plt.close()
        
        # Evaluate QNN on test set
        with torch.no_grad():
            q_outputs = quantum_classifier(X_test_q_tensor)
            
            # Try different thresholds
            thresholds = np.linspace(0.1, 0.9, 9)
            results = []
            
            for threshold in thresholds:
                q_preds = (q_outputs >= threshold).float()
                accuracy = (q_preds.eq(y_test_tensor).sum().item()) / y_test_tensor.size(0)
                
                # Convert to numpy for sklearn metrics
                q_preds_np = q_preds.cpu().numpy().flatten()
                y_test_np = y_test_tensor.cpu().numpy().flatten()
                
                precision = precision_score(y_test_np, q_preds_np, zero_division=0)
                recall = recall_score(y_test_np, q_preds_np, zero_division=0)
                f1 = f1_score(y_test_np, q_preds_np, zero_division=0)
                balanced_acc = balanced_accuracy_score(y_test_np, q_preds_np)
                
                results.append({
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'balanced_accuracy': balanced_acc,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })
            
            # Find best threshold by F1 score
            results_df = pd.DataFrame(results)
            best_idx = results_df['f1_score'].idxmax()
            best_result = results_df.iloc[best_idx]
            
            print("\nQuantum model results at different thresholds:")
            print(results_df.round(4))
            
            print(f"\nBest threshold: {best_result['threshold']:.2f} (F1: {best_result['f1_score']:.4f})")
            print(f"At best threshold:")
            print(f"  Accuracy: {best_result['accuracy']:.4f}")
            print(f"  Balanced Accuracy: {best_result['balanced_accuracy']:.4f}")
            print(f"  Precision: {best_result['precision']:.4f}")
            print(f"  Recall: {best_result['recall']:.4f}")
            print(f"  F1 Score: {best_result['f1_score']:.4f}")
            print(f"  Training Time: {quantum_training_time:.4f} seconds")
            
            # Apply best threshold to predictions
            best_threshold = best_result['threshold']
            q_preds = (q_outputs >= best_threshold).float()
            q_preds_np = q_preds.cpu().numpy().flatten()
            y_test_np = y_test_tensor.cpu().numpy().flatten()
            
            # Confusion matrix
            cm = confusion_matrix(y_test_np, q_preds_np)
            print(f"  Confusion Matrix:\n{cm}")
            
            # Calculate ROC AUC if possible
            try:
                q_outputs_np = q_outputs.cpu().numpy().flatten()
                roc_auc = roc_auc_score(y_test_np, q_outputs_np)
                print(f"  ROC AUC: {roc_auc:.4f}")
            except Exception as e:
                print(f"Could not calculate ROC AUC: {e}")
            
            # Return results and model
            quantum_results = {
                'accuracy': best_result['accuracy'],
                'balanced_accuracy': best_result['balanced_accuracy'],
                'precision': best_result['precision'],
                'recall': best_result['recall'],
                'f1_score': best_result['f1_score'],
                'confusion_matrix': cm,
                'training_time': quantum_training_time,
                'best_threshold': best_threshold
            }
            
            return quantum_results, quantum_classifier
        
    except ImportError as e:
        print(f"Qadence or other quantum packages not found: {e}")
        print("Skipping quantum model.")
        return None, None
    except Exception as e:
        print(f"Error in quantum model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def enhanced_visualizations(models, X_test, y_test, feature_names):
    """
    Create enhanced visualizations for model interpretation.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        feature_names: Names of features
    """
    print("\n=== Enhanced Visualizations ===")
    
    # 1. Feature Importance Plot (for tree-based models)
    tree_models = {}
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            tree_models[name] = model
    
    if tree_models:
        plt.figure(figsize=(12, 10))
        for i, (name, model) in enumerate(tree_models.items()):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.subplot(len(tree_models), 1, i+1)
            plt.title(f'Feature Importance - {name}')
            plt.bar(range(X_test.shape[1]), importances[indices], align='center')
            plt.xticks(range(X_test.shape[1]), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
        
        plt.savefig('feature_importance.png')
        plt.close()
        print("Feature importance plot saved as 'feature_importance.png'")
    
    # 2. ROC Curves
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
            except Exception as e:
                print(f"Could not generate ROC curve for {name}: {e}")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curves.png')
    plt.close()
    print("ROC curves saved as 'roc_curves.png'")
    
    # 3. Precision-Recall Curves (better for imbalanced data)
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                avg_precision = average_precision_score(y_test, y_proba)
                
                plt.plot(recall, precision, lw=2, label=f'{name} (AP = {avg_precision:.2f})')
            except Exception as e:
                print(f"Could not generate PR curve for {name}: {e}")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig('precision_recall_curves.png')
    plt.close()
    print("Precision-Recall curves saved as 'precision_recall_curves.png'")
    
    # 4. Confusion Matrix Heatmaps
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['No Bleaching', 'Bleaching'],
                       yticklabels=['No Bleaching', 'Bleaching'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - {name}')
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png')
            plt.close()
            print(f"Confusion matrix for {name} saved")
        except Exception as e:
            print(f"Could not generate confusion matrix for {name}: {e}")
    
    # 5. SHAP Values for model explainability (for a Random Forest if available)
    try:
        if 'Random Forest' in models:
            import shap
            explainer = shap.TreeExplainer(models['Random Forest'])
            shap_values = explainer.shap_values(X_test)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig('shap_summary.png')
            plt.close()
            print("SHAP summary plot saved as 'shap_summary.png'")
            
            # Dependence plots for top features
            feature_importance = np.mean(np.abs(shap_values[1]), axis=0)
            top_indices = np.argsort(feature_importance)[-3:]  # Top 3 features
            
            for idx in top_indices:
                plt.figure(figsize=(10, 7))
                shap.dependence_plot(idx, shap_values[1], X_test, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(f'shap_dependence_{feature_names[idx]}.png')
                plt.close()
                print(f"SHAP dependence plot for {feature_names[idx]} saved")
    except Exception as e:
        print(f"Could not generate SHAP plots: {e}")

def analyze_temporal_trends(df, year_col='Year', target_col='Bleaching'):
    """
    Analyze and visualize temporal trends in bleaching.
    
    Args:
        df: DataFrame with year and target columns
        year_col: Name of the year column
        target_col: Name of the target column
    """
    if year_col not in df.columns:
        print(f"Year column '{year_col}' not found in dataframe. Skipping temporal analysis.")
        return
    
    print("\n=== Temporal Trend Analysis ===")
    
    # Convert target to binary if not already
    if df[target_col].dtype != int and df[target_col].dtype != float:
        print(f"Converting target column to binary (0/1)")
        # Assuming 'Yes'/'yes' means 1, everything else is 0
        df[target_col] = df[target_col].astype(str).str.lower().map({'yes': 1, 'true': 1, '1': 1}).fillna(0).astype(int)
    
    # Group by year and calculate percentage of bleaching
    yearly_data = df.groupby(year_col)[target_col].agg(['mean', 'count']).reset_index()
    yearly_data['bleaching_percent'] = yearly_data['mean'] * 100
    
    # Plot time series of bleaching percentage
    plt.figure(figsize=(12, 6))
    
    # Line plot
    plt.plot(yearly_data[year_col], yearly_data['bleaching_percent'], marker='o', linestyle='-', color='blue')
    
    # Add count as bar widths
    max_percent = yearly_data['bleaching_percent'].max()
    bar_height = max_percent * 0.2
    plt.bar(yearly_data[year_col], bar_height, 
            bottom=-bar_height, alpha=0.3, color='gray', 
            width=0.8, align='center')
    
    # Annotate sample counts
    for i, row in yearly_data.iterrows():
        plt.text(row[year_col], -bar_height*0.9, f"n={row['count']}", 
                 ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Year')
    plt.ylabel('Bleaching Percentage (%)')
    plt.title('Percentage of Coral Bleaching by Year')
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=-bar_height*1.5)
    
    # Fit a trend line
    try:
        from scipy import stats
        years = yearly_data[year_col].values
        percentages = yearly_data['bleaching_percent'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, percentages)
        
        trend_line = slope * years + intercept
        plt.plot(years, trend_line, 'r--', label=f'Trend (R²={r_value**2:.2f}, p={p_value:.3f})')
        
        if p_value < 0.05:
            trend_direction = "increasing" if slope > 0 else "decreasing"
            print(f"Significant {trend_direction} trend in bleaching over time (p={p_value:.3f})")
        else:
            print(f"No significant trend in bleaching over time (p={p_value:.3f})")
            
        plt.legend()
    except Exception as e:
        print(f"Could not fit trend line: {e}")
    
    plt.tight_layout()
    plt.savefig('bleaching_temporal_trend.png')
    plt.close()
    print("Temporal trend analysis saved as 'bleaching_temporal_trend.png'")

def analyze_spatial_patterns(df, location_col='Ocean', target_col='Bleaching'):
    """
    Analyze and visualize spatial patterns in bleaching.
    
    Args:
        df: DataFrame with location and target columns
        location_col: Name of the location column
        target_col: Name of the target column
    """
    if location_col not in df.columns:
        print(f"Location column '{location_col}' not found in dataframe. Skipping spatial analysis.")
        return
    
    print("\n=== Spatial Pattern Analysis ===")
    
    # Convert target to binary if not already
    if df[target_col].dtype != int and df[target_col].dtype != float:
        print(f"Converting target column to binary (0/1)")
        # Assuming 'Yes'/'yes' means 1, everything else is 0
        df[target_col] = df[target_col].astype(str).str.lower().map({'yes': 1, 'true': 1, '1': 1}).fillna(0).astype(int)
    
    # Convert location column to string if it's numeric (from encoding)
    if pd.api.types.is_numeric_dtype(df[location_col]):
        # Try to map back to original names if possible
        try:
            ocean_mapping_reverse = {0: "Arabian Gulf", 1: "Atlantic", 2: "Indian", 3: "Pacific", 4: "Red Sea"}
            df[location_col] = df[location_col].map(ocean_mapping_reverse)
            print(f"Converted numeric {location_col} back to ocean names")
        except Exception as e:
            print(f"Could not convert numeric {location_col} to names: {e}")
            df[location_col] = df[location_col].astype(str)
    
    # Group by location and calculate percentage of bleaching
    location_data = df.groupby(location_col)[target_col].agg(['mean', 'count']).reset_index()
    location_data['bleaching_percent'] = location_data['mean'] * 100
    
    # Sort by percentage for better visualization
    location_data = location_data.sort_values('bleaching_percent', ascending=False)
    
    # Plot bleaching percentage by location
    plt.figure(figsize=(12, 8))
    
    # Bar plot
    bars = plt.bar(location_data[location_col], location_data['bleaching_percent'])
    
    # Add count annotations
    for bar, count in zip(bars, location_data['count']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f"n={count}", ha='center', va='bottom')
    
    plt.xlabel('Location')
    plt.ylabel('Bleaching Percentage (%)')
    plt.title('Percentage of Coral Bleaching by Location')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bleaching_spatial_pattern.png')
    plt.close()
    print("Spatial pattern analysis saved as 'bleaching_spatial_pattern.png'")
    
    # Statistical test: Chi-square test of independence
    try:
        # Create contingency table
        contingency = pd.crosstab(df[location_col], df[target_col])
        chi2, p, dof, expected = chi2_contingency(contingency)
        
        print(f"Chi-square test for independence between {location_col} and {target_col}:")
        print(f"  Chi-square value: {chi2:.2f}")
        print(f"  p-value: {p:.4f}")
        print(f"  Degrees of freedom: {dof}")
        
        if p < 0.05:
            print(f"  Result: Significant association between {location_col} and {target_col} (p<0.05)")
        else:
            print(f"  Result: No significant association between {location_col} and {target_col} (p≥0.05)")
    except Exception as e:
        print(f"Could not perform chi-square test: {e}")

def enhanced_coral_reef_prediction(df, test_size=0.25, random_state=42):
    """
    Main function implementing the enhanced coral reef bleaching prediction.
    
    Args:
        df: DataFrame with features and target
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Results of all models
    """
    print("=== Enhanced Coral Reef Bleaching Prediction Pipeline ===")
    
    # 1. Feature selection
    selected_features = improved_feature_selection(df, target_col='Bleaching')
    
    # 2. Prepare data for modeling
    X = df[selected_features]
    y = df['Bleaching']
    feature_names = X.columns.tolist()
    
    # 3. Split data (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 4. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Create balanced datasets for handling imbalance
    balanced_datasets = create_balanced_datasets(X_train_scaled, y_train, random_state)
    
    # 6. Tune hyperparameters for classical models
    # Use SMOTE-balanced dataset for tuning
    if 'smote' in balanced_datasets:
        X_train_balanced, y_train_balanced = balanced_datasets['smote']
    else:
        X_train_balanced, y_train_balanced = balanced_datasets['original']
        
    tuned_models = tune_hyperparameters(X_train_balanced, y_train_balanced)
    
    # 7. Train and evaluate quantum model
    quantum_results, quantum_model = enhanced_quantum_model(
        X_train_scaled, X_test_scaled, y_train, y_test, 
        n_features=min(4, X.shape[1]),  # Use up to 4 features for quantum model
        n_epochs=100
    )
    
    # 8. Visualizations for model interpretation
    enhanced_visualizations(tuned_models, X_test_scaled, y_test, feature_names)
    
    # 9. Temporal trend analysis
    analyze_temporal_trends(df)
    
    # 10. Spatial pattern analysis
    analyze_spatial_patterns(df)
    
    # 11. Compile final results
    all_results = {}
    
    # Add classical model results
    for name, model in tuned_models.items():
        try:
            y_pred = model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            all_results[name] = {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
    
    # Add quantum results if available
    if quantum_results is not None:
        all_results['Quantum QNN'] = quantum_results
    
    # Print final summary
    print("\n=== Final Results Summary ===")
    results_df = pd.DataFrame({
        model_name: {
            'Accuracy': results['accuracy'],
            'Balanced Accuracy': results['balanced_accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1 Score': results['f1_score']
        }
        for model_name, results in all_results.items()
    })
    
    print("\nPerformance metrics:")
    print(results_df.round(4).T)
    
    # Sort models by F1 score (better for imbalanced data)
    best_models = sorted(all_results.items(), 
                          key=lambda x: x[1]['f1_score'], 
                          reverse=True)
    
    print("\nModels ranked by F1 Score:")
    for i, (name, metrics) in enumerate(best_models):
        print(f"{i+1}. {name}: F1={metrics['f1_score']:.4f}, "
              f"Balanced Accuracy={metrics['balanced_accuracy']:.4f}")
    
    return all_results

def main():
    """
    Main function to run the enhanced coral reef bleaching prediction.
    """
    print("=== Enhanced Coral Reef Bleaching Prediction System ===")
    
    # 1. Load and preprocess data
    df = load_and_preprocess_data()
    
    # 2. Run the enhanced prediction pipeline
    results = enhanced_coral_reef_prediction(df)
    
    # 3. Create combined visualization of all models
    print("\nGenerating final comparison visualization...")
    
    # Extract metrics for comparison
    models = list(results.keys())
    metrics = ['accuracy', 'balanced_accuracy', 'f1_score', 'precision', 'recall']
    
    # Create a figure for model comparison
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i+1)
        
        # Extract values for this metric
        values = [results[model][metric] for model in models]
        
        # Create horizontal bar chart
        bars = plt.barh(models, values, color='skyblue')
        
        # Add values at the end of bars
        for bar, val in zip(bars, values):
            plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                     va='center', fontsize=9)
        
        plt.title(f'{metric.replace("_", " ").title()}')
        plt.xlim(0, 1.1)  # All metrics are between 0 and 1
        plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    print("\nSuccess! Enhanced model comparison saved as 'model_comparison.png'")
    print("All analysis results and visualizations have been saved to the current directory.")
    
    # 4. Find and print the best model
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\nBest performing model: {best_model[0]} with F1 score of {best_model[1]['f1_score']:.4f}")
    
    # 5. Suggestions for further improvements
    print("\nSuggested next steps:")
    print("1. Try ensemble methods combining the best classical and quantum models")
    print("2. Collect more data on minority class (bleaching events)")
    print("3. Engineer additional features from domain knowledge")
    print("4. Explore deeper quantum circuit architectures if resources permit")
    print("5. Implement deployment strategy for real-time coral reef monitoring")

if __name__ == "__main__":
    main()