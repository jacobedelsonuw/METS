"""
Data Preprocessing Module for Metabolic Syndrome Research

This module provides functionality to preprocess clinical and wearable data
for metabolic syndrome research.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import json
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_preprocessing")

class DataPreprocessor:
    """
    Class for preprocessing clinical and wearable data for metabolic syndrome research.
    """
    
    def __init__(self, config=None):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        # Default configuration
        self.config = {
            'imputation_method': 'knn',  # Options: 'mean', 'median', 'knn', 'mice'
            'scaling_method': 'standard',  # Options: 'standard', 'minmax', 'robust', 'none'
            'outlier_detection': 'iqr',  # Options: 'iqr', 'zscore', 'isolation_forest', 'none'
            'categorical_encoding': 'onehot',  # Options: 'onehot', 'label', 'target', 'none'
            'feature_selection': 'none',  # Options: 'variance', 'correlation', 'rfe', 'none'
            'missing_threshold': 0.3,  # Maximum proportion of missing values allowed
            'correlation_threshold': 0.8,  # Threshold for correlation-based feature selection
            'variance_threshold': 0.01,  # Threshold for variance-based feature selection
            'random_state': 42,
            'output_dir': 'data'
        }
        
        # Update with user-provided configuration
        if config:
            self.config.update(config)
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        logger.info("DataPreprocessor initialized with configuration:")
        for key, value in self.config.items():
            logger.info(f"  {key}: {value}")
    
    def preprocess_data(self, data):
        """
        Preprocess data for metabolic syndrome analysis.
        
        Args:
            data: DataFrame with clinical and/or wearable data
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Preprocessing data with {len(data)} rows and {len(data.columns)} columns")
        
        # Make a copy of the data
        processed_data = data.copy()
        
        # Step 1: Handle missing values
        processed_data = self._handle_missing_values(processed_data)
        
        # Step 2: Handle outliers
        processed_data = self._handle_outliers(processed_data)
        
        # Step 3: Encode categorical variables
        processed_data = self._encode_categorical_variables(processed_data)
        
        # Step 4: Scale numerical variables
        processed_data = self._scale_numerical_variables(processed_data)
        
        # Step 5: Feature selection (if enabled)
        processed_data = self._select_features(processed_data)
        
        # Step 6: Create derived features
        processed_data = self._create_derived_features(processed_data)
        
        logger.info(f"Preprocessing completed: {len(processed_data)} rows, {len(processed_data.columns)} columns")
        
        return processed_data
    
    def _handle_missing_values(self, data):
        """
        Handle missing values in the data.
        
        Args:
            data: DataFrame with missing values
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values")
        
        # Make a copy of the data
        result = data.copy()
        
        # Check missing values
        missing_summary = result.isnull().sum()
        missing_percent = missing_summary / len(result) * 100
        
        # Log columns with missing values
        columns_with_missing = missing_summary[missing_summary > 0]
        if not columns_with_missing.empty:
            logger.info(f"Columns with missing values:")
            for col, count in columns_with_missing.items():
                logger.info(f"  {col}: {count} values ({missing_percent[col]:.1f}%)")
        else:
            logger.info("No missing values found")
            return result
        
        # Drop columns with too many missing values
        columns_to_drop = missing_percent[missing_percent > self.config['missing_threshold'] * 100].index.tolist()
        if columns_to_drop:
            logger.info(f"Dropping columns with >={self.config['missing_threshold'] * 100:.1f}% missing values: {columns_to_drop}")
            result = result.drop(columns=columns_to_drop)
        
        # Separate numerical and categorical columns
        numerical_cols = result.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = result.select_dtypes(include=['object', 'category']).columns
        
        # Handle missing values based on configuration
        if self.config['imputation_method'] == 'mean':
            # Mean imputation for numerical columns
            if not numerical_cols.empty:
                imputer = SimpleImputer(strategy='mean')
                result[numerical_cols] = imputer.fit_transform(result[numerical_cols])
                logger.info(f"Applied mean imputation to {len(numerical_cols)} numerical columns")
            
            # Mode imputation for categorical columns
            if not categorical_cols.empty:
                imputer = SimpleImputer(strategy='most_frequent')
                result[categorical_cols] = imputer.fit_transform(result[categorical_cols])
                logger.info(f"Applied mode imputation to {len(categorical_cols)} categorical columns")
        
        elif self.config['imputation_method'] == 'median':
            # Median imputation for numerical columns
            if not numerical_cols.empty:
                imputer = SimpleImputer(strategy='median')
                result[numerical_cols] = imputer.fit_transform(result[numerical_cols])
                logger.info(f"Applied median imputation to {len(numerical_cols)} numerical columns")
            
            # Mode imputation for categorical columns
            if not categorical_cols.empty:
                imputer = SimpleImputer(strategy='most_frequent')
                result[categorical_cols] = imputer.fit_transform(result[categorical_cols])
                logger.info(f"Applied mode imputation to {len(categorical_cols)} categorical columns")
        
        elif self.config['imputation_method'] == 'knn':
            # KNN imputation for numerical columns
            if not numerical_cols.empty:
                try:
                    imputer = KNNImputer(n_neighbors=5)
                    result[numerical_cols] = imputer.fit_transform(result[numerical_cols])
                    logger.info(f"Applied KNN imputation to {len(numerical_cols)} numerical columns")
                except Exception as e:
                    logger.error(f"Error applying KNN imputation: {str(e)}")
                    logger.info("Falling back to median imputation for numerical columns")
                    imputer = SimpleImputer(strategy='median')
                    result[numerical_cols] = imputer.fit_transform(result[numerical_cols])
            
            # Mode imputation for categorical columns
            if not categorical_cols.empty:
                imputer = SimpleImputer(strategy='most_frequent')
                result[categorical_cols] = imputer.fit_transform(result[categorical_cols])
                logger.info(f"Applied mode imputation to {len(categorical_cols)} categorical columns")
        
        elif self.config['imputation_method'] == 'mice':
            # MICE imputation for numerical columns
            if not numerical_cols.empty:
                try:
                    imputer = IterativeImputer(max_iter=10, random_state=self.config['random_state'])
                    result[numerical_cols] = imputer.fit_transform(result[numerical_cols])
                    logger.info(f"Applied MICE imputation to {len(numerical_cols)} numerical columns")
                except Exception as e:
                    logger.error(f"Error applying MICE imputation: {str(e)}")
                    logger.info("Falling back to median imputation for numerical columns")
                    imputer = SimpleImputer(strategy='median')
                    result[numerical_cols] = imputer.fit_transform(result[numerical_cols])
            
            # Mode imputation for categorical columns
            if not categorical_cols.empty:
                imputer = SimpleImputer(strategy='most_frequent')
                result[categorical_cols] = imputer.fit_transform(result[categorical_cols])
                logger.info(f"Applied mode imputation to {len(categorical_cols)} categorical columns")
        
        else:
            logger.warning(f"Unknown imputation method: {self.config['imputation_method']}")
            logger.info("No imputation applied")
        
        return result
    
    def _handle_outliers(self, data):
        """
        Handle outliers in the data.
        
        Args:
            data: DataFrame with outliers
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info("Handling outliers")
        
        # Make a copy of the data
        result = data.copy()
        
        # Skip if outlier detection is disabled
        if self.config['outlier_detection'] == 'none':
            logger.info("Outlier detection disabled")
            return result
        
        # Get numerical columns
        numerical_cols = result.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Exclude ID columns and target variables
        exclude_cols = ['patient_id', 'id', 'has_metabolic_syndrome', 'metabolic_criteria_count']
        exclude_cols.extend([col for col in numerical_cols if col.startswith('meets_')])
        
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        logger.info(f"Checking {len(numerical_cols)} numerical columns for outliers")
        
        # Handle outliers based on configuration
        if self.config['outlier_detection'] == 'iqr':
            # IQR method
            for col in numerical_cols:
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Count outliers
                outliers = ((result[col] < lower_bound) | (result[col] > upper_bound)).sum()
                
                if outliers > 0:
                    logger.info(f"Capping {outliers} outliers in {col} using IQR method")
                    
                    # Cap outliers
                    result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif self.config['outlier_detection'] == 'zscore':
            # Z-score method
            for col in numerical_cols:
                mean = result[col].mean()
                std = result[col].std()
                z_scores = (result[col] - mean) / std
                
                # Count outliers (|z| > 3)
                outliers = (abs(z_scores) > 3).sum()
                
                if outliers > 0:
                    logger.info(f"Capping {outliers} outliers in {col} using Z-score method")
                    
                    # Cap outliers
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                    result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif self.config['outlier_detection'] == 'isolation_forest':
            # Isolation Forest method
            try:
                # Select subset of numerical columns for outlier detection
                outlier_cols = [col for col in numerical_cols if result[col].nunique() > 10]
                
                if outlier_cols:
                    # Create and fit Isolation Forest
                    iso_forest = IsolationForest(
                        contamination=0.05,  # Assume 5% outliers
                        random_state=self.config['random_state']
                    )
                    outlier_labels = iso_forest.fit_predict(result[outlier_cols])
                    
                    # Count outliers
                    outliers = (outlier_labels == -1).sum()
                    
                    if outliers > 0:
                        logger.info(f"Detected {outliers} outliers using Isolation Forest")
                        
                        # Create outlier flag
                        result['outlier'] = (outlier_labels == -1).astype(int)
                        
                        # For each column, replace outliers with median
                        for col in outlier_cols:
                            median_value = result[col].median()
                            result.loc[result['outlier'] == 1, col] = median_value
                        
                        # Drop outlier flag
                        result = result.drop(columns=['outlier'])
            except Exception as e:
                logger.error(f"Error using Isolation Forest: {str(e)}")
                logger.info("Falling back to IQR method")
                
                # Fall back to IQR method
                for col in numerical_cols:
                    q1 = result[col].quantile(0.25)
                    q3 = result[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Count outliers
                    outliers = ((result[col] < lower_bound) | (result[col] > upper_bound)).sum()
                    
                    if outliers > 0:
                        logger.info(f"Capping {outliers} outliers in {col} using IQR method")
                        
                        # Cap outliers
                        result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
        
        else:
            logger.warning(f"Unknown outlier detection method: {self.config['outlier_detection']}")
            logger.info("No outlier handling applied")
        
        return result
    
    def _encode_categorical_variables(self, data):
        """
        Encode categorical variables in the data.
        
        Args:
            data: DataFrame with categorical variables
            
        Returns:
            DataFrame with encoded categorical variables
        """
        logger.info("Encoding categorical variables")
        
        # Make a copy of the data
        result = data.copy()
        
        # Skip if categorical encoding is disabled
        if self.config['categorical_encoding'] == 'none':
            logger.info("Categorical encoding disabled")
            return result
        
        # Get categorical columns
        categorical_cols = result.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Exclude ID columns
        exclude_cols = [<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>