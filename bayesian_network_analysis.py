"""
Bayesian Network Analysis for Metabolic Syndrome

This script implements advanced Bayesian network analysis for metabolic syndrome
using the pgmpy library. It builds on the knowledge representation model to identify
probabilistic relationships between metabolic parameters and provides inference
capabilities for risk assessment and clinical decision support.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime
import json
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import pgmpy modules
from pgmpy.models import BayesianNetwork, BayesianModel
from pgmpy.estimators import (
    HillClimbSearch, 
    K2Score, 
    BDeuScore, 
    BicScore, 
    MaximumLikelihoodEstimator, 
    BayesianEstimator
)
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling

# Create directories for Bayesian network files
os.makedirs("bayesian_network", exist_ok=True)
os.makedirs("bayesian_network/models", exist_ok=True)
os.makedirs("bayesian_network/visualizations", exist_ok=True)
os.makedirs("bayesian_network/results", exist_ok=True)

# --- 1. Data Preprocessing for Bayesian Network Analysis ---
def preprocess_data_for_bn(measurements_file, demographics_file, conditions_file=None, medications_file=None):
    """
    Preprocess data for Bayesian network analysis.
    
    Parameters:
    -----------
    measurements_file : str
        Path to the measurements CSV file
    demographics_file : str
        Path to the demographics CSV file
    conditions_file : str, optional
        Path to the conditions CSV file
    medications_file : str, optional
        Path to the medications CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data for Bayesian network analysis
    """
    print("Preprocessing data for Bayesian network analysis...")
    
    # Check if files exist
    if not os.path.exists(measurements_file):
        print(f"Measurements file {measurements_file} not found.")
        return None
    
    if not os.path.exists(demographics_file):
        print(f"Demographics file {demographics_file} not found.")
        return None
    
    try:
        # Load data
        measurements_df = pd.read_csv(measurements_file)
        demographics_df = pd.read_csv(demographics_file)
        
        # Load conditions and medications if available
        conditions_df = None
        if conditions_file and os.path.exists(conditions_file):
            conditions_df = pd.read_csv(conditions_file)
        
        medications_df = None
        if medications_file and os.path.exists(medications_file):
            medications_df = pd.read_csv(medications_file)
        
        # Define concept IDs for metabolic syndrome components
        metabolic_concept_ids = {
            'waist_circumference': [3007070, 3047181],
            'bmi': [3020891],
            'systolic_bp': [3012888, 3013742],
            'diastolic_bp': [3028288, 3013682],
            'hdl_cholesterol': [3004501, 3001420],
            'triglycerides': [3022192, 3016723],
            'fasting_glucose': [3004249, 3023103],
            'hba1c': [3019550],
            'total_cholesterol': [3006906],
            'ldl_cholesterol': [3015632],
            'insulin': [3009744],
            'crp': [3034639],
            'alt': [3010813],
            'ast': [3006923],
            'ggt': [3024561],
            'creatinine': [3024128],
            'albumin': [3016502]
        }
        
        # Filter measurements for relevant concept IDs
        all_concept_ids = [id for ids in metabolic_concept_ids.values() for id in ids]
        filtered_measurements = measurements_df[measurements_df['measurement_concept_id'].isin(all_concept_ids)]
        
        # Get the most recent measurement for each person and concept
        most_recent = filtered_measurements.sort_values('measurement_datetime').groupby(['person_id', 'measurement_concept_id']).last().reset_index()
        
        # Create features for each measurement type
        features_df = pd.DataFrame({'person_id': most_recent['person_id'].unique()})
        
        for feature, concept_ids in metabolic_concept_ids.items():
            # Filter measurements for this feature
            feature_measurements = most_recent[most_recent['measurement_concept_id'].isin(concept_ids)]
            
            if not feature_measurements.empty:
                # Pivot to get one row per person
                feature_pivot = feature_measurements.pivot_table(
                    index='person_id',
                    columns='measurement_concept_id',
                    values='value_as_number',
                    aggfunc='mean'
                ).reset_index()
                
                # Combine multiple concept IDs for the same feature
                if len(concept_ids) > 1:
                    # Try each concept ID in order of preference
                    for concept_id in concept_ids:
                        if concept_id in feature_pivot.columns:
                            feature_pivot[feature] = feature_pivot[concept_id]
                            break
                else:
                    # Only one concept ID
                    concept_id = concept_ids[0]
                    if concept_id in feature_pivot.columns:
                        feature_pivot[feature] = feature_pivot[concept_id]
                
                # Keep only person_id and the new feature column
                if feature in feature_pivot.columns:
                    feature_pivot = feature_pivot[['person_id', feature]]
                    
                    # Merge with features dataframe
                    features_df = features_df.merge(feature_pivot, on='person_id', how='left')
        
        # Merge with demographics
        demographics_subset = demographics_df[['person_id', 'age', 'gender', 'race', 'ethnicity']]
        merged_df = features_df.merge(demographics_subset, on='person_id', how='left')
        
        # Add condition indicators if available
        if conditions_df is not None:
            # Define condition concept IDs of interest
            condition_concept_ids = {
                'diabetes': [201826],
                'hypertension': [4311629],
                'hyperlipidemia': [432867],
                'obesity': [4324893],
                'insulin_resistance': [4030518],
                'metabolic_syndrome': [4185932]
            }
            
            # Create binary indicators for each condition
            for condition, concept_ids in condition_concept_ids.items():
                condition_filter = conditions_df['condition_concept_id'].isin(concept_ids)
                condition_persons = conditions_df[condition_filter]['person_id'].unique()
                merged_df[condition] = merged_df['person_id'].isin(condition_persons).astype(int)
        
        # Add medication indicators if available
        if medications_df is not None:
            # Define medication concept IDs of interest
            medication_concept_ids = {
                'antidiabetic': [1529331, 1525215, 1592673, 1597756, 1560171, 40239216, 1583722],
                'antihypertensive': [1340128, 1332418, 1373225, 1346686, 1319998],
                'lipid_lowering': [1545958, 1551860, 1549686, 1510813, 1526475]
            }
            
            # Create binary indicators for each medication
            for medication, concept_ids in medication_concept_ids.items():
                medication_filter = medications_df['drug_concept_id'].isin(concept_ids)
                medication_persons = medications_df[medication_filter]['person_id'].unique()
                merged_df[medication] = merged_df['person_id'].isin(medication_persons).astype(int)
        
        # Identify metabolic syndrome based on standard criteria
        merged_df['has_abdominal_obesity'] = np.where(
            (merged_df['gender'] == 'Male') & (merged_df['waist_circumference'] >= 102) |
            (merged_df['gender'] == 'Female') & (merged_df['waist_circumference'] >= 88),
            1, 0
        )
        
        merged_df['has_elevated_triglycerides'] = np.where(
            merged_df['triglycerides'] >= 150,
            1, 0
        )
        
        merged_df['has_reduced_hdl'] = np.where(
            (merged_df['gender'] == 'Male') & (merged_df['hdl_cholesterol'] < 40) |
            (merged_df['gender'] == 'Female') & (merged_df['hdl_cholesterol'] < 50),
            1, 0
        )
        
        merged_df['has_hypertension'] = np.where(
            (merged_df['systolic_bp'] >= 130) | (merged_df['diastolic_bp'] >= 85),
            1, 0
        )
        
        merged_df['has_hyperglycemia'] = np.where(
            (merged_df['fasting_glucose'] >= 100) | (merged_df['hba1c'] >= 5.7),
            1, 0
        )
        
        # Calculate total criteria met
        criteria_columns = [
            'has_abdominal_obesity',
            'has_elevated_triglycerides',
            'has_reduced_hdl',
            'has_hypertension',
            'has_hyperglycemia'
        ]
        
        merged_df['criteria_count'] = merged_df[criteria_columns].sum(axis=1)
        
        # Define metabolic syndrome (3 or more criteria)
        merged_df['has_metabolic_syndrome'] = np.where(
            merged_df['criteria_count'] >= 3,
            1, 0
        )
        
        # Fill missing values
        numeric_columns = merged_df.select_dtypes(include=['float64', 'int64']).columns
        merged_df[numeric_columns] = merged_df[numeric_columns].fillna(merged_df[numeric_columns].median())
        
        categorical_columns = merged_df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            merged_df[col] = merged_df[col].fillna(merged_df[col].mode()[0])
        
        # Discretize continuous variables for Bayesian network
        discretized_df = merged_df.copy()
        
        # Define columns to discretize
        continuous_columns = [
            'waist_circumference', 'bmi', 'systolic_bp', 'diastolic_bp',
            'hdl_cholesterol', 'triglycerides', 'fasting_glucose', 'hba1c',
            'total_cholesterol', 'ldl_cholesterol', 'age'
        ]
        
        # Discretize each column
        for col in continuous_columns:
            if col in discretized_df.columns:
                discretized_df[col] = pd.qcut(
                    discretized_df[col], 
                    q=5, 
                    labels=['very_low', 'low', 'medium', 'high', 'very_high'],
                    duplicates='drop'
                )
        
        # Convert categorical columns to string
        for col in discretized_df.columns:
            if discretized_df[col].dtype.name == 'category':
                discretized_df[col] = discretized_df[col].astype(str)
        
        # Save preprocessed data
        merged_df.to_csv("bayesian_network/preprocessed_data_continuous.csv", index=False)
        discretized_df.to_csv("bayesian_network/preprocessed_data_discrete.csv", index=False)
        
        print(f"Preprocessed data saved to bayesian_network/preprocessed_data_continuous.csv and bayesian_network/preprocessed_data_discrete.csv")
        
        return discretized_df
    
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None

# --- 2. Bayesian Network Structure Learning ---
def learn_bn_structure(data, algorithm='hc', score='k2', max_indegree=4, black_list=None, white_list=None):
    """
    Learn the structure of a Bayesian network from data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Preprocessed data for Bayesian network analysis
    algorithm : str
        Structure learning algorithm ('hc' for Hill-Climbing)
    score : str
        Scoring function ('k2', 'bdeu', or 'bic')
    max_indegree : int
        Maximum number of parents for each node
    black_list : list
        List of edges to exclude from the network
    white_list : list
        List of edges to include in the network
        
    Returns:
    --------
    pgmpy.models.BayesianNetwork
        Learned Bayesian network structure
    """
    print(f"Learning Bayesian network structure using {algorithm} algorithm with {score} score...")
    
    try:
        # Select columns for structure learning
        # Exclude person_id and redundant columns
        exclude_cols = ['person_id', 'criteria_count']
        selected_cols = [col for col in data.columns if col not in exclude_cols]
        selected_data = data[selected_cols]
        
        # Initialize scoring function
        if score == 'k2':
            score_func = K2Score(selected_data)
        elif score == 'bdeu':
            score_func = BDeuScore(selected_data)
        elif score == 'bic':
            score_func = BicScore(selected_data)
        else:
            print(f"Unknown score function: {score}. Using K2Score.")
            score_func = K2Score(selected_data)
        
        # Initialize structure learning algorithm
        if algorithm == 'hc':
            hc = HillClimbSearch(selected_data)
            
            # Learn structure
            model = hc.estimate(
                scoring_method=score_func,
                max_indegree=max_indegree,
                black_list=black_list,
                white_list=white_list
            )
            
            # Convert to BayesianNetwork
            bn_model = BayesianNetwork(model.edges())
            
            # Save edges to file
            edges_df = pd.DataFrame(model.edges(), columns=['from', 'to'])
            edges_df.to_csv(f"bayesian_network/models/bn_edges_{algorithm}_{score}.csv", index=False)
            
            print(f"Learned {len(model.edges())} edges.")
            print(f"Edges saved to bayesian_network/models/bn_edges_{algorithm}_{score}.csv")
            
            return bn_model
        
        else:
            print(f"Unknown algorithm: {algorithm}")
            return None
    
    except Exception as e:
        print(f"Error learning Bayesian network structure: {e}")
        return None

# --- 3. Bayesian Network Parameter Learning ---
def learn_bn_parameters(model, data, method='mle'):
    """
    Learn the parameters of a Bayesian network from data.
    
    Parameters:
    -----------
    model : pgmpy.models.BayesianNetwork
        Bayesian network structure
    data : pandas.DataFrame
        Preprocessed data for Bayesian network analysis
    method : str
        Parameter learning method ('mle' for Maximum Likelihood Estimation,
        'bayesian' for Bayesian Estimation)
        
    Returns:
    --------
    pgmpy.models.BayesianNetwork
        Bayesian network with learned parameters
    """
    print(f"Learning Bayesian network parameters using {method} method...")
    
    try:
        # Select columns for parameter learning
        # Exclude person_id and redundant columns
        exclude_cols = ['person_id', 'criteria_count']
        selected_cols = [col for col in data.columns if col not in exclude_cols]
        selected_data = data[selected_cols]
        
        # Check if all model nodes are in the data
        for node in model.nodes():
            if node not in selected_data.columns:
                print(f"Node {node} not found in data columns.")
                return None
        
        # Learn parameters
        if method == 'mle':
            model.fit(selected_data, estimator=MaximumLikelihoodEstimator)
        elif method == 'bayesian':
            model.fit(selected_data, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=5)
        else:
            print(f"Unknown parameter learning method: {method}. Using MLE.")
            model.fit(selected_data, estimator=MaximumLikelihoodEst<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>