"""
AllOfUS Data Extraction for Metabolic Syndrome Research

This module implements comprehensive data extraction and processing from the AllOfUS
Research Program database for metabolic syndrome analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import json
import warnings
import requests
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("allofus_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("allofus_extraction")

# Create output directories
DATA_DIR = os.path.join(os.getcwd(), "data")
PROCESSED_DATA_DIR = os.path.join(os.getcwd(), "processed_data")
FIGURES_DIR = os.path.join(os.getcwd(), "figures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

class AllOfUSDataExtractor:
    """
    Class for extracting and processing data from the AllOfUS Research Program database.
    """
    
    def __init__(self, config=None):
        """
        Initialize the AllOfUS data extractor.
        
        Args:
            config: Configuration dictionary with extraction parameters
        """
        # Default configuration
        self.config = {
            'api_base_url': 'https://workbench.researchallofus.org/api/v1',
            'cohort_size': 10000,  # Number of participants to include in cohort
            'random_seed': 42,
            'age_range': [18, 85],  # Age range for inclusion
            'include_demographics': True,
            'include_physical_measurements': True,
            'include_lab_results': True,
            'include_medications': True,
            'include_conditions': True,
            'include_wearables': True,
            'missing_value_threshold': 0.3,  # Maximum proportion of missing values allowed
            'output_format': 'csv'  # Options: 'csv', 'parquet', 'json'
        }
        
        # Update with user-provided configuration
        if config:
            self.config.update(config)
        
        # Initialize data containers
        self.demographics_data = None
        self.physical_measurements_data = None
        self.lab_results_data = None
        self.medications_data = None
        self.conditions_data = None
        self.wearables_data = None
        self.integrated_data = None
        self.participant_ids = None
        
        # Initialize API connection
        self.api_client = None
        
        logger.info("AllOfUSDataExtractor initialized with configuration:")
        for key, value in self.config.items():
            logger.info(f"  {key}: {value}")
    
    def connect_to_api(self, api_key=None):
        """
        Connect to the AllOfUS API.
        
        Args:
            api_key: API key for authentication (optional)
            
        Returns:
            Boolean indicating connection success
        """
        logger.info("Connecting to AllOfUS API")
        
        # In a real implementation, this would establish a connection to the AllOfUS API
        # For this example, we'll simulate a successful connection
        self.api_client = {
            'connected': True,
            'api_key': api_key,
            'base_url': self.config['api_base_url']
        }
        
        logger.info("Successfully connected to AllOfUS API")
        
        return True
    
    def define_cohort(self, cohort_criteria=None):
        """
        Define a cohort of participants based on criteria.
        
        Args:
            cohort_criteria: Dictionary with cohort selection criteria (optional)
            
        Returns:
            List of participant IDs in the cohort
        """
        logger.info("Defining participant cohort")
        
        # Default cohort criteria
        default_criteria = {
            'age_min': self.config['age_range'][0],
            'age_max': self.config['age_range'][1],
            'include_sexes': ['MALE', 'FEMALE'],
            'include_races': ['WHITE', 'BLACK', 'ASIAN', 'HISPANIC', 'OTHER'],
            'exclude_conditions': [],
            'require_physical_measurements': True,
            'require_lab_results': True
        }
        
        # Update with user-provided criteria
        if cohort_criteria:
            default_criteria.update(cohort_criteria)
        
        # In a real implementation, this would query the AllOfUS API to get a cohort
        # For this example, we'll generate synthetic participant IDs
        np.random.seed(self.config['random_seed'])
        participant_ids = [f"P{i:08d}" for i in range(1, self.config['cohort_size'] + 1)]
        
        self.participant_ids = participant_ids
        
        logger.info(f"Defined cohort with {len(participant_ids)} participants")
        
        return participant_ids
    
    def extract_demographics(self):
        """
        Extract demographic data for the cohort.
        
        Returns:
            DataFrame with demographic data
        """
        if not self.participant_ids:
            logger.error("No participant cohort defined")
            return None
        
        logger.info("Extracting demographic data")
        
        # In a real implementation, this would query the AllOfUS API
        # For this example, we'll generate synthetic demographic data
        
        # Generate data
        demographics = []
        
        for participant_id in self.participant_ids:
            # Generate random demographic data
            age = np.random.randint(self.config['age_range'][0], self.config['age_range'][1] + 1)
            sex = np.random.choice(['MALE', 'FEMALE'])
            race = np.random.choice(['WHITE', 'BLACK', 'ASIAN', 'HISPANIC', 'OTHER'], 
                                   p=[0.6, 0.15, 0.1, 0.1, 0.05])
            
            # Education level
            education = np.random.choice([
                'LESS_THAN_HIGH_SCHOOL', 'HIGH_SCHOOL', 'SOME_COLLEGE', 
                'COLLEGE_GRADUATE', 'ADVANCED_DEGREE'
            ], p=[0.1, 0.25, 0.3, 0.25, 0.1])
            
            # Income level
            income = np.random.choice([
                'LESS_THAN_25K', '25K_TO_50K', '50K_TO_75K', 
                '75K_TO_100K', 'GREATER_THAN_100K'
            ], p=[0.2, 0.25, 0.25, 0.15, 0.15])
            
            # Add to demographics list
            demographics.append({
                'participant_id': participant_id,
                'age': age,
                'sex': sex,
                'race': race,
                'education': education,
                'income': income
            })
        
        # Convert to DataFrame
        demographics_df = pd.DataFrame(demographics)
        
        # Store demographics data
        self.demographics_data = demographics_df
        
        logger.info(f"Extracted demographic data for {len(demographics_df)} participants")
        
        return demographics_df
    
    def extract_physical_measurements(self):
        """
        Extract physical measurement data for the cohort.
        
        Returns:
            DataFrame with physical measurement data
        """
        if not self.participant_ids:
            logger.error("No participant cohort defined")
            return None
        
        logger.info("Extracting physical measurement data")
        
        # In a real implementation, this would query the AllOfUS API
        # For this example, we'll generate synthetic physical measurement data
        
        # Generate data
        measurements = []
        
        for participant_id in self.participant_ids:
            # Get demographic data if available
            if self.demographics_data is not None:
                participant_demo = self.demographics_data[self.demographics_data['participant_id'] == participant_id]
                if not participant_demo.empty:
                    age = participant_demo['age'].values[0]
                    sex = participant_demo['sex'].values[0]
                else:
                    age = np.random.randint(self.config['age_range'][0], self.config['age_range'][1] + 1)
                    sex = np.random.choice(['MALE', 'FEMALE'])
            else:
                age = np.random.randint(self.config['age_range'][0], self.config['age_range'][1] + 1)
                sex = np.random.choice(['MALE', 'FEMALE'])
            
            # Generate random physical measurements
            # Height in cm (influenced by sex)
            if sex == 'MALE':
                height = np.random.normal(175, 7)
            else:
                height = np.random.normal(162, 6)
            
            # Weight in kg (influenced by age and sex)
            if sex == 'MALE':
                base_weight = 70 + (age - 30) * 0.1
            else:
                base_weight = 60 + (age - 30) * 0.1
            
            weight = np.random.normal(base_weight, base_weight * 0.15)
            
            # BMI
            bmi = weight / ((height / 100) ** 2)
            
            # Waist circumference in cm (influenced by sex and BMI)
            if sex == 'MALE':
                waist = 80 + (bmi - 25) * 2 + np.random.normal(0, 5)
            else:
                waist = 75 + (bmi - 25) * 2 + np.random.normal(0, 5)
            
            # Hip circumference in cm (influenced by sex and BMI)
            if sex == 'MALE':
                hip = 90 + (bmi - 25) * 2 + np.random.normal(0, 5)
            else:
                hip = 95 + (bmi - 25) * 2 + np.random.normal(0, 5)
            
            # Waist-to-hip ratio
            whr = waist / hip
            
            # Blood pressure (influenced by age and BMI)
            systolic_bp = 110 + (age - 30) * 0.5 + (bmi - 25) * 1 + np.random.normal(0, 8)
            diastolic_bp = 70 + (age - 30) * 0.2 + (bmi - 25) * 0.5 + np.random.normal(0, 5)
            
            # Heart rate (influenced by age and BMI)
            heart_rate = 70 + (age - 30) * 0.1 + (bmi - 25) * 0.2 + np.random.normal(0, 8)
            
            # Measurement date
            measurement_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
            
            # Add to measurements list
            measurements.append({
                'participant_id': participant_id,
                'measurement_date': measurement_date,
                'height_cm': height,
                'weight_kg': weight,
                'bmi': bmi,
                'waist_circumference_cm': waist,
                'hip_circumference_cm': hip,
                'waist_hip_ratio': whr,
                'systolic_bp_mmHg': systolic_bp,
                'diastolic_bp_mmHg': diastolic_bp,
                'heart_rate_bpm': heart_rate
            })
        
        # Convert to DataFrame
        measurements_df = pd.DataFrame(measurements)
        
        # Store physical measurements data
        self.physical_measurements_data = measurements_df
        
        logger.info(f"Extracted physical measurement data for {len(measurements_df)} participants")
        
        return measurements_df
    
    def extract_lab_results(self):
        """
        Extract laboratory result data for the cohort.
        
        Returns:
            DataFrame with laboratory result data
        """
        if not self.participant_ids:
            logger.error("No participant cohort defined")
            return None
        
        logger.info("Extracting laboratory result data")
        
        # In a real implementation, this would query the AllOfUS API
        # For this example, we'll generate synthetic lab result data
        
        # Generate data
        lab_results = []
        
        for participant_id in self.participant_ids:
            # Get physical measurement data if available
            if self.physical_measurements_data is not None:
                participant_meas = self.physical_measurements_data[self.physical_measurements_data['participant_id'] == participant_id]
                if not participant_meas.empty:
                    bmi = participant_meas['bmi'].values[0]
                    measurement_date = participant_meas['measurement_date'].values[0]
                else:
                    bmi = np.random.normal(27, 5)
                    measurement_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
            else:
                bmi = np.random.normal(27, 5)
                measurement_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
            
            # Lab date (close to measurement date)
            lab_date = measurement_date + timedelta(days=np.random.randint(-30, 30))
            
            # Generate random lab results (influenced by BMI)
            # Lipid panel
            total_cholesterol = 180 + (bmi - 25) * 2 + np.random.normal(0, 20)
            hdl_cholesterol = 50 - (bmi - 25) * 0.5 + np.random.normal(0, 8)
            ldl_cholesterol = 110 + (bmi - 25) * 1.5 + np.random.normal(0, 15)
            triglycerides = 120 + (bmi - 25) * 3 + np.random.normal(0, 30)
            
            # Glucose metabolism
            fasting_glucose = 90 + (bmi - 25) * 1 + np.random.normal(0, 10)
            hba1c = 5.5 + (bmi - 25) * 0.05 + np.random.normal(0, 0.3)
            insulin = 10 + (bmi - 25) * 0.5 + np.random.normal(0, 3)
            
            # Liver function
            alt = 20 + (bmi - 25) * 0.5 + np.random.normal(0, 5)
            ast = 20 + (bmi - 25) * 0.4 + np.random.normal(0, 5)
            
            # Kidney function
            creatinine = 0.9 + np.random.normal(0, 0.1)
            egfr = 90 + np.random.normal(0, 10)
            
            # Inflammation
            crp = 1 + (bmi - 25) * 0.1 + np.random.normal(0, 0.5)
            
            # Add to lab results list
            lab_results.append({
                'participant_id': participant_id,
                'lab_date': lab_date,
                'total_cholesterol_mg_dL': total_cholesterol,
                'hdl_cholesterol_mg_dL': hdl_cholesterol,
                'ldl_cholesterol_mg_dL': ldl_cholesterol,
                'triglycerides_mg_dL': triglycerides,
                'fasting_glucose_mg_dL': fasting_glucose,
                'hba1c_percent': hba1c,
                'insulin_uIU_mL': insulin,
                'alt_U_L': alt,
                'ast_U_L': ast,
                'creatinine_mg_dL': creatinine,
                'egfr_mL_min': egfr,
                'crp_mg_L': crp
            })
        
        # Convert to DataFrame
        lab_results_df = pd.DataFrame(lab_results)
        
        # Store lab results data
        self.lab_results_data = lab_results_df
        
        logger.info(f"Extracted laboratory result data for {len(lab_results_df)} participants")
        
        return lab_results_df
    
    def extract_medications(self):
        """
        Extract medication data for the cohort.
        
        Returns:
            DataFrame with medication data
        """
        if not self.participant_ids:
            logger.error("No participant cohort defined")
            return None
        
        logger.info("Extracting medication data")
        
        # In a real implementation, this would query the AllOfUS API
        # For this example, we'll generate synthetic medication data
        
        # Define common medications for metabolic syndrome
        antihypertensives = [
            'Lisinopril', 'Amlodipine', 'Losartan', 'Hydrochlorothiazide', 
            'Metoprolol', 'Valsartan', 'Atenolol'
        ]
        
        lipid_lowering = [
            'Atorvastatin', 'Simvastatin', 'Rosuvastatin', 'Pravastatin',