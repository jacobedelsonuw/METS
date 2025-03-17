import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import json
import datetime
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import sys
import base64
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("risk_calculator_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("risk_calculator_app")

# Create output directory
OUTPUT_DIR = os.path.join(os.getcwd(), "risk_calculator_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MetabolicSyndromeRiskCalculator:
    """
    Class for calculating metabolic syndrome risk based on clinical and wearable data.
    """
    
    def __init__(self, model_path: Optional[str] = None, knowledge_graph_path: Optional[str] = None):
        """
        Initialize the risk calculator.
        
        Args:
            model_path: Path to the risk calculator model file
            knowledge_graph_path: Path to the knowledge graph file
        """
        # Define metabolic syndrome criteria
        self.metabolic_syndrome_criteria = {
            "waist_circumference_male": 102,  # cm
            "waist_circumference_female": 88,  # cm
            "triglycerides": 150,  # mg/dL
            "hdl_cholesterol_male": 40,  # mg/dL
            "hdl_cholesterol_female": 50,  # mg/dL
            "systolic_bp": 130,  # mmHg
            "diastolic_bp": 85,  # mmHg
            "fasting_glucose": 100  # mg/dL
        }
        
        # Define wearable data thresholds
        self.wearable_thresholds = {
            "daily_steps": {
                "low_risk": 10000,
                "moderate_risk": 7000,
                "high_risk": 5000
            },
            "active_minutes": {
                "low_risk": 30,
                "moderate_risk": 20,
                "high_risk": 10
            },
            "sleep_duration": {
                "low_risk": 8,
                "moderate_risk": 7,
                "high_risk": 6
            },
            "resting_heart_rate": {
                "low_risk": 60,
                "moderate_risk": 70,
                "high_risk": 80
            },
            "heart_rate_variability": {
                "low_risk": 50,
                "moderate_risk": 30,
                "high_risk": 20
            },
            "glucose_variability": {
                "low_risk": 10,
                "moderate_risk": 20,
                "high_risk": 30
            },
            "time_in_range": {
                "low_risk": 90,
                "moderate_risk": 70,
                "high_risk": 50
            },
            "stress_score": {
                "low_risk": 25,
                "moderate_risk": 50,
                "high_risk": 75
            }
        }
        
        # Define risk categories
        self.risk_categories = {
            "low": (0, 25),
            "moderate": (25, 50),
            "high": (50, 75),
            "very_high": (75, 100)
        }
        
        # Define metabolic syndrome subtypes
        self.metabolic_syndrome_subtypes = {
            "Obesity-Dominant": {
                "description": "Characterized by central obesity as the primary feature",
                "clinical_features": ["increased_waist_circumference", "elevated_bmi", "insulin_resistance"],
                "wearable_features": ["low_daily_steps", "low_active_minutes", "high_resting_heart_rate"],
                "recommendations": [
                    "Focus on caloric restriction and portion control",
                    "Aim for at least 10,000 steps daily",
                    "Include resistance training 2-3 times per week",
                    "Consider consultation with a registered dietitian",
                    "Monitor waist circumference weekly"
                ]
            },
            "Dyslipidemia-Dominant": {
                "description": "Characterized by abnormal lipid levels as the primary feature",
                "clinical_features": ["elevated_triglycerides", "reduced_hdl_cholesterol", "elevated_ldl_cholesterol"],
                "wearable_features": ["variable_daily_steps", "variable_active_minutes", "moderate_glucose_variability"],
                "recommendations": [
                    "Reduce saturated and trans fat intake",
                    "Increase omega-3 fatty acid consumption",
                    "Engage in regular aerobic exercise (30+ minutes, 5 days/week)",
                    "Consider plant sterols/stanols supplementation",
                    "Monitor lipid profile every 3-6 months"
                ]
            },
            "Hypertension-Dominant": {
                "description": "Characterized by elevated blood pressure as the primary feature",
                "clinical_features": ["elevated_systolic_bp", "elevated_diastolic_bp", "elevated_pulse_pressure"],
                "wearable_features": ["high_resting_heart_rate", "low_heart_rate_variability", "high_stress_score"],
                "recommendations": [
                    "Reduce sodium intake to <2,300 mg/day",
                    "Adopt DASH diet principles",
                    "Practice stress reduction techniques daily",
                    "Monitor blood pressure regularly",
                    "Ensure adequate potassium intake"
                ]
            },
            "Hyperglycemia-Dominant": {
                "description": "Characterized by elevated glucose levels as the primary feature",
                "clinical_features": ["elevated_fasting_glucose", "elevated_hba1c", "insulin_resistance"],
                "wearable_features": ["high_glucose_variability", "low_time_in_range", "high_resting_heart_rate"],
                "recommendations": [
                    "Reduce simple carbohydrate intake",
                    "Eat smaller, more frequent meals",
                    "Exercise within 30 minutes of high-carb meals",
                    "Consider continuous glucose monitoring",
                    "Ensure adequate fiber intake (25-30g daily)"
                ]
            },
            "Mixed/Balanced": {
                "description": "Characterized by a balanced presentation of multiple metabolic syndrome components",
                "clinical_features": ["moderate_waist_circumference", "moderate_triglycerides", "moderate_blood_pressure", "moderate_fasting_glucose"],
                "wearable_features": ["moderate_daily_steps", "moderate_active_minutes", "moderate_heart_rate_variability"],
                "recommendations": [
                    "Follow Mediterranean diet principles",
                    "Aim for 150+ minutes of moderate exercise weekly",
                    "Ensure 7-8 hours of quality sleep nightly",
                    "Practice stress management techniques",
                    "Regular monitoring of all metabolic parameters"
                ]
            }
        }
        
        # Load model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info(f"Loaded risk calculator model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading risk calculator model: {str(e)}")
        
        # Load knowledge graph if provided
        self.knowledge_graph = None
        if knowledge_graph_path and os.path.exists(knowledge_graph_path):
            try:
                self.knowledge_graph = nx.read_graphml(knowledge_graph_path)
                logger.info(f"Loaded knowledge graph from {knowledge_graph_path}")
            except Exception as e:
                logger.error(f"Error loading knowledge graph: {str(e)}")
        
        # Initialize feature importances
        self.feature_importances = {
            "waist_circumference": 0.15,
            "triglycerides": 0.12,
            "hdl_cholesterol": 0.10,
            "systolic_bp": 0.10,
            "diastolic_bp": 0.08,
            "fasting_glucose": 0.12,
            "daily_steps": 0.08,
            "active_minutes": 0.06,
            "sleep_duration": 0.05,
            "resting_heart_rate": 0.05,
            "heart_rate_variability": 0.04,
            "glucose_variability": 0.05
        }
        
        # Initialize model type
        self.model_type = "Ensemble (Random Forest + Gradient Boosting)"
    
    def calculate_clinical_risk_score(self, clinical_data: Dict[str, float], gender: str = "Male") -> float:
        """
        Calculate metabolic syndrome risk score based on clinical data.
        
        Args:
            clinical_data: Dictionary of clinical measurements
            gender: Gender of the patient ("Male" or "Female")
        
        Returns:
            Risk score (0-100)
        """
        # Initialize risk score
        risk_score = 0
        
        # Check waist circumference
        if "waist_circumference" in clinical_data:
            waist_threshold = self.metabolic_syndrome_criteria["waist_circumference_male"] if gender == "Male" else self.metabolic_syndrome_criteria["waist_circumference_female"]
            if clinical_data["waist_circumference"] >= waist_threshold:
                risk_score += 20
            elif clinical_data["waist_circumference"] >= waist_threshold * 0.9:
                risk_score += 10
        
        # Check triglycerides
        if "triglycerides" in clinical_data:
            if clinical_data["triglycerides"] >= self.metabolic_syndrome_criteria["triglycerides"]:
                risk_score += 20
            elif clinical_data["triglycerides"] >= self.metabolic_syndrome_criteria["triglycerides"] * 0.8:
                risk_score += 10
        
        # Check HDL cholesterol
        if "hdl_cholesterol" in clinical_data:
            hdl_threshold = self.metabolic_syndrome_criteria["hdl_cholesterol_male"] if gender == "Male" else self.metabolic_syndrome_criteria["hdl_cholesterol_female"]
            if clinical_data["hdl_cholesterol"] <= hdl_threshold:
                risk_score += 20
            elif clinical_data["hdl_cholesterol"] <= hdl_threshold * 1.2:
                risk_score += 10
        
        # Check blood pressure
        if "systolic_bp" in clinical_data and "diastolic_bp" in clinical_data:
            if (clinical_data["systolic_bp"] >= self.metabolic_syndrome_criteria["systolic_bp"] or 
                clinical_data["diastolic_bp"] >= self.metabolic_syndrome_criteria["diastolic_bp"]):
                risk_score += 20
            elif (clinical_data["systolic_bp"] >= self.metabolic_syndrome_criteria["systolic_bp"] * 0.9 or 
                  clinical_data["diastolic_bp"] >= self.metabolic_syndrome_criteria["diastolic_bp"] * 0.9):
                risk_score += 10
        
        # Check fasting glucose
        if "fasting_glucose" in clinical_data:
            if clinical_data["fasting_glucose"] >= self.metabolic_syndrome_criteria["fasting_glucose"]:
                risk_score += 20
            elif clinical_data["fasting_glucose"] >= self.metabolic_syndrome_criteria["fasting_glucose"] * 0.9:
                risk_score += 10
        
        # Normalize risk score to 0-100 scale
        normalized_score = min(risk_score, 100)
        
        return normalized_score
    
    def calculate_wearable_risk_score(self, wearable_data: Dict[str, float]) -> float:
        """
        Calculate metabolic syndrome risk score based on wearable data.
        
        Args:
            wearable_data: Dictionary of wearable measurements
        
        Returns:
            Risk score (0-100)
        """
        # Initialize risk score
        risk_score = 0
        
        # Check daily steps
        if "daily_steps" in wearable_data:
            if wearable_data["daily_steps"] <= self.wearable_thresholds["daily_steps"]["high_risk"]:
                risk_score += 15
            elif wearable_data["daily_steps"] <= self.wearable_thresholds["daily_steps"]["moderate_risk"]:
                risk_score += 7.5
        
        # Check active minutes
        if "active_minutes" in wearable_data:
            if wearable_data["active_minutes"] <= self.wearable_thresholds["active_minutes"]["high_risk"]:
                risk_score += 15
            elif wearable_data["active_minutes"] <= self.wearable_thresholds["active_minutes"]["moderate_risk"]:
                risk_score += 7.5
        
        # Check sleep duration
        if "sleep_duration" in wearable_data:
            if wearable_data["sleep_duration"] <= self.wearable_thresholds["sleep_duration"]["high_risk"]:
                risk_score += 15
            elif wearable_data["sleep_duration"] <= self.wearable_thresholds["sleep_duration"]["moderate_risk"]:
                risk_score += 7.5
        
        # Check resting heart rate
        if "resting_heart_rate" in wearable_data:
            if wearable_data["resting_heart_rate"] >= self.wearable_thresholds["resting_heart_rate"]["high_risk"]:
                risk_score += 15
            elif wearable_data["resting_heart_rate"] >= self.wearable_thresholds["resting_heart_rate"]["moderate_risk"]:
                risk_score += 7.5
        
        # Check heart rate variability
        if "heart_rate_variability" in wearable_data:
            if wearable_data["heart_rate_variability"] <= self.wearable_thresholds["heart_rate_variability"]["high_risk"]:
                risk_score += 10
            elif wearable_data["heart_rate_variability"] <= self.wearable_thresholds["heart_rate_variability"]["moderate_risk"]:
                risk_score += 5
        
        # Check glucose variability
        if "glucose_variability" in wearable_data:
            if wearable_data["glucose_variability"] >= self.wearable_thresholds["glucose_variability"]["high_risk"]:
                risk_score += 15
            elif wearable_data["glucose_variability"] >= self.wearable_thresholds["glucose_variability"]["moderate_risk"]:
                risk_score += 7.5
        
        # Check time in range
        if "time_in_range" in wearable_data:
            if wearable_data["time_in_range"] <= self.wearable_thresholds["time_in_range"]["high_risk"]:
                risk_score += 15
            elif wearable_data["time_in_range"] <= self.wearable_thresholds["time_in_range"]["moderate_risk"]:
                risk_score += 7.5
        
        # Normalize risk score to 0-100 scale
        normalized_score = min(risk_score, 100)
        
        return normalized_score
    
    def calculate_combined_risk_score(self, clinical_data: Dict[str, float], wearable_data: Dict[str, float], gender: str = "Male") -> float:
        """
        Calculate combined metabolic syndrome risk score based on clinical and wearable data.
        
        Args:
            clinical_data: Dictionary of clinical measurements
            wearable_data: Dictionary of wearable measurements
            gender: Gender of the patient ("Male" or "Female")
        
        Returns:
            Combined risk score (0-100)
        """
        # Calculate individual risk scores
        clinical_score = self.calculate_clinical_risk_score(clinical_data, gender)
        wearable_score = self.calculate_wearable_risk_score(wearable_data)
        
        # Calculate combined score (weighted average)
        # Clinical data has higher weight (70%) than wearable data (30%)
        combined_score = 0.7 * clinical_score + 0.3 * wearable_score
        
        return combined_score
    
    def determine_risk_category(self, risk_score: float) -> str:
        """
        Determine risk category based on risk score.
        
        Args:
            risk_score: Risk score (0-100)
        
        Returns:
            Risk category ("low", "moderate", "high", or "very_high")
        """
        for category, (lower, upper) in self.risk_categories.items():
            if lower <= risk_score < upper:
                return category
        
        return "very_high"  # Default to very high if score is 100
    
    def determine_metabolic_syndrome_subtype(self, clinical_data: Dict[str, float], wearable_data: Dict[str, float], gender: str = "Male") -> str:
        """
        Determine the metabolic syndrome subtype based on clinical and wearable data.
        
        Args:
            clinical_data: Dictionary of clinical measurements
            wearable_data: Dictionary of wearable measurements
            gender: Gender of the patient ("Male" or "Female")
        
        Returns:
            Metabolic syndrome subtype
        """
        # Initialize scores for each subtype
        subtype_scores = {
            "Obesity-Dominant": 0,
            "Dyslipidemia-Dominant": 0,
            "Hypertension-Dominant": 0,
            "Hyperglycemia-Dominant": 0,
            "Mixed/Balanced": 0
        }
        
        # Score based on clinical data
        
        # Obesity-Dominant indicators
        if "waist_circumference" in clinical_data:
            waist_threshold = self.metabolic_syndrome_criteria["waist_circumference_male"] if gender == "Male" else self.metabolic_syndrome_criteria["waist_circumference_female"]
            if clinical_data["waist_circumference"] > waist_threshold + 10:
                subtype_scores["Obesity-Dominant"] += 3
            elif clinical_data["waist_circumference"] > waist_threshold:
                subtype_scores["Obesity-Dominant"] += 2
        
        if "bmi" in clinical_data:
            if clinical_data["bmi"] > 35:
                subtype_scores["Obesity-Dominant"] += 3
            elif clinical_data["bmi"] > 30:
                subtype_scores["Obesity-Dominant"] += 2
            elif clinical_data["bmi"] > 25:
                subtype_scores["Obesity-Dominant"] += 1
        
        # Dyslipidemia-Dominant indicators
        if "triglycerides" in clinical_data:
            if clinical_data["triglycerides"] > 200:
                subtype_scores["Dyslipidemia-Dominant"] += 3
            elif clinical_data["triglycerides"] > self.metabolic_syndrome_criteria["triglycerides"]:
                subtype_scores["Dyslipidemia-Dominant"] += 2
        
        if "hdl_cholesterol" in clinical_data:
            hdl_threshold = self.metabolic_syndrome_criteria["hdl_cholesterol_male"] if gender == "Male" else self.metabolic_syndrome_criteria["hdl_cholesterol_female"]
            if clinical_data["hdl_cholesterol"] < hdl_threshold - 10:
                subtype_scores["Dyslipidemia-Dominant"] += 3
            elif clinical_data["hdl_cholesterol"] < hdl_threshold:
                subtype_scores["Dyslipidemia-Dominant"] += 2
        
        if "ldl_cholesterol" in clinical_data:
            if clinical_data["ldl_cholesterol"] > 160:
                subtype_scores["Dyslipidemia-Dominant"] += 3
            elif clinical_data["ldl_cholesterol"] > 130:
                subtype_scores["Dyslipidemia-Dominant"] += 2
            elif clinical_data["ldl_cholesterol"] > 100:
                subtype_scores["Dyslipidemia-Dominant"] += 1
        
        # Hypertension-Dominant indicators
        if "systolic_bp" in clinical_data:
            if clinical_data["systolic_bp"] > 160:
                subtype_scores["Hypertension-Dominant"] += 3
            elif clinical_data["systolic_bp"] > 140:
                subtype_scores["Hypertension-Dominant"] += 2
            elif clinical_data["systolic_bp"] > self.metabolic_syndrome_criteria["systolic_bp"]:
                subtype_scores["Hypertension-Dominant"] += 1
        
        if "diastolic_bp" in clinical_data:
            if clinical_data["diastolic_bp"] > 100:
                subtype_scores["Hypertension-Dominant"] += 3
            elif clinical_data["diastolic_bp"] > 90:
                subtype_scores["Hypertension-Dominant"] += 2
            elif clinical_data["diastolic_bp"] > self.metabolic_syndrome_criteria["diastolic_bp"]:
                subtype_scores["Hypertension-Dominant"] += 1
        
        # Hyperglycemia-Dominant indicators
        if "fasting_glucose" in clinical_data:
            if clinical_data["fasting_glucose"] > 126:
                subtype_scores["Hyperglycemia-Dominant"] += 3
            elif clinical_data["fasting_glucose"] > 110:
                subtype_scores["Hyperglycemia-Dominant"] += 2
            elif clinical_data["fasting_glucose"] > self.metabolic_syndrome_criteria["fasting_glucose"]:
                subtype_scores["Hyperglycemia-Dominant"] += 1
        
        if "hba1c" in clinical_data:
            if clinical_data["hba1c"] > 6.5:
                subtype_scores["Hyperglycemia-Dominant"] += 3
            elif clinical_data["hba1c"] > 6.0:
                subtype_scores["Hyperglycemia-Dominant"] += 2
            elif clinical_data["hba1c"] > 5.7:
                subtype_scores["Hyperglycemia-Dominant"] += 1
        
        # Score based on wearable data
        
        # Obesity-Dominant indicators
        if "daily_steps" in wearable_data:
            if wearable_data["daily_steps"] < 5000:
                subtype_scores["Obesity-Dominant"] += 2
            elif wearable_data["daily_steps"] < 7000:
                subtype_scores["Obesity-Dominant"] += 1
        
        if "active_minutes" in wearable_data:
            if wearable_data["active_minutes"] < 10:
                subtype_scores["Obesity-Dominant"] += 2
            elif wearable_data["active_minutes"] < 20:
                subtype_scores["Obesity-Dominant"] += 1
        
        # Hypertension-Dominant indicators
        if "resting_heart_rate" in wearable_data:
            if wearable_data["resting_heart_rate"] > 80:
                subtype_scores["Hypertension-Dominant"] += 2
            elif wearable_data["resting_heart_rate"] > 70:
                subtype_scores["Hypertension-Dominant"] += 1
        
        if "heart_rate_variability" in wearable_data:
            if wearable_data["heart_rate_variability"] < 20:
                subtype_scores["Hypertension-Dominant"] += 2
            elif wearable_data["heart_rate_variability"] < 30:
                subtype_scores["Hypertension-Dominant"] += 1
        
        if "stress_score" in wearable_data:
            if wearable_data["stress_score"] > 75:
                subtype_scores["Hypertension-Dominant"] += 2
            elif wearable_data["stress_score"] > 50:
                subtype_scores["Hypertension-Dominant"] += 1
        
        # Hyperglycemia-Dominant indicators
        if "glucose_variability" in wearable_data:
            if wearable_data["glucose_variability"] > 30:
                subtype_scores["Hyperglycemia-Dominant"] += 2
            elif wearable_data["glucose_variability"] > 20:
                subtype_scores["Hyperglycemia-Dominant"] += 1
        
        if "time_in_range" in wearable_data:
            if wearable_data["time_in_range"] < 50:
                subtype_scores["Hyperglycemia-Dominant"] += 2
            elif wearable_data["time_in_range"] < 70:
                subtype_scores["Hyperglycemia-Dominant"] += 1
        
        # Calculate Mixed/Balanced score
        # If no subtype has a significantly higher score, increase Mixed/Balanced score
        max_score = max(subtype_scores.values())
        if max_score < 3:
            subtype_scores["Mixed/Balanced"] += 3
        
        # Get the subtype with the highest score
        max_subtype = max(subtype_scores, key=subtype_scores.get)
        
        return max_subtype
    
    def generate_recommendations(self, risk_category: str, subtype: str) -> List[str]:
        """
        Generate personalized recommendations based on risk category and metabolic syndrome subtype.
        
        Args:
            risk_category: Risk category ("low", "moderate", "high", or "very_high")
            subtype: Metabolic syndrome subtype
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Add general recommendations based on risk category
        if risk_category == "low":
            recommendations.extend([
                "Maintain current healthy lifestyle habits",
                "Continue regular physical activity (150+ minutes/week)",
                "Follow a balanced diet rich in fruits, vegetables, and whole grains",
                "Schedule annual check-ups with your healthcare provider"
            ])
        
        elif risk_category == "moderate":
            recommendations.extend([
                "Increase physical activity to 150-300 minutes/week",
                "Reduce processed food and added sugar intake",
                "Monitor weight and waist circumference monthly",
                "Schedule check-ups every 6-12 months with your healthcare provider"
            ])
        
        elif risk_category == "high":
            recommendations.extend([
                "Consult with healthcare provider about metabolic syndrome risk",
                "Increase physical activity to 300+ minutes/week",
                "Follow a structured diet plan (Mediterranean or DASH diet recommended)",
                "Monitor blood pressure, glucose, and lipids regularly",
                "Schedule check-ups every 3-6 months with your healthcare provider"
            ])
        
        elif risk_category == "very_high":
            recommendations.extend([
                "Urgent consultation with healthcare provider about metabolic syndrome",
                "Consider structured weight management program",
                "Follow a medically supervised exercise program",
                "Monitor blood pressure, glucose, and lipids weekly",
                "Schedule monthly check-ups with your healthcare provider"
            ])
        
        # Add subtype-specific recommendations
        if subtype in self.metabolic_syndrome_subtypes:
            subtype_recommendations = self.metabolic_syndrome_subtypes[subtype]["recommendations"]
            recommendations.extend(subtype_recommendations)
        
        return recommendations
    
    def generate_risk_report(self, clinical_data: Dict[str, float], wearable_data: Dict[str, float], gender: str = "Male") -> Dict[str, Any]:
        """
        Generate a comprehensive risk report based on clinical and wearable data.
        
        Args:
            clinical_data: Dictionary of clinical measurements
            wearable_data: Dictionary of wearable measurements
            gender: Gender of the patient ("Male" or "Female")
        
        Returns:
            Dictionary containing risk assessment results
        """
        # Calculate risk scores
        clinical_score = self.calculate_clinical_risk_score(clinical_data, gender)
        wearable_score = self.calculate_wearable_risk_score(wearable_data)
        combined_score = self.calculate_combined_risk_score(clinical_data, wearable_data, gender)
        
        # Determine risk category
        risk_category = self.determine_risk_category(combined_score)
        
        # Determine metabolic syndrome subtype
        subtype = self.determine_metabolic_syndrome_subtype(clinical_data, wearable_data, gender)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(risk_category, subtype)
        
        # Calculate component scores
        component_scores = {
            "Obesity": 0,
            "Dyslipidemia": 0,
            "Hypertension": 0,
            "Hyperglycemia": 0,
            "Physical Inactivity": 0
        }
        
        # Obesity component
        if "waist_circumference" in clinical_data:
            waist_threshold = self.metabolic_syndrome_criteria["waist_circumference_male"] if gender == "Male" else self.metabolic_syndrome_criteria["waist_circumference_female"]
            if clinical_data["waist_circumference"] >= waist_threshold:
                component_scores["Obesity"] += 50
            elif clinical_data["waist_circumference"] >= waist_threshold * 0.9:
                component_scores["Obesity"] += 25
        
        if "bmi" in clinical_data:
            if clinical_data["bmi"] >= 30:
                component_scores["Obesity"] += 50
            elif clinical_data["bmi"] >= 25:
                component_scores["Obesity"] += 25
        
        # Dyslipidemia component
        if "triglycerides" in clinical_data:
            if clinical_data["triglycerides"] >= self.metabolic_syndrome_criteria["triglycerides"]:
                component_scores["Dyslipidemia"] += 50
            elif clinical_data["triglycerides"] >= self.metabolic_syndrome_criteria["triglycerides"] * 0.8:
                component_scores["Dyslipidemia"] += 25
        
        if "hdl_cholesterol" in clinical_data:
            hdl_threshold = self.metabolic_syndrome_criteria["hdl_cholesterol_male"] if gender == "Male" else self.metabolic_syndrome_criteria["hdl_cholesterol_female"]
            if clinical_data["hdl_cholesterol"] <= hdl_threshold:
                component_scores["Dyslipidemia"] += 50
            elif clinical_data["hdl_cholesterol"] <= hdl_threshold * 1.2:
                component_scores["Dyslipidemia"] += 25
        
        # Hypertension component
        if "systolic_bp" in clinical_data:
            if clinical_data["systolic_bp"] >= self.metabolic_syndrome_criteria["systolic_bp"]:
                component_scores["Hypertension"] += 50
            elif clinical_data["systolic_bp"] >= self.metabolic_syndrome_criteria["systolic_bp"] * 0.9:
                component_scores["Hypertension"] += 25
        
        if "diastolic_bp" in clinical_data:
            if clinical_data["diastolic_bp"] >= self.metabolic_syndrome_criteria["diastolic_bp"]:
                component_scores["Hypertension"] += 50
            elif clinical_data["diastolic_bp"] >= self.metabolic_syndrome_criteria["diastolic_bp"] * 0.9:
                component_scores["Hypertension"] += 25
        
        # Hyperglycemia component
        if "fasting_glucose" in clinical_data:
            if clinical_data["fasting_glucose"] >= self.metabolic_syndrome_criteria["fasting_glucose"]:
                component_scores["Hyperglycemia"] += 50
            elif clinical_data["fasting_glucose"] >= self.metabolic_syndrome_criteria["fasting_glucose"] * 0.9:
                component_scores["Hyperglycemia"] += 25
        
        if "hba1c" in clinical_data:
            if clinical_data["hba1c"] >= 5.7:
                component_scores["Hyperglycemia"] += 50
            elif clinical_data["hba1c"] >= 5.5:
                component_scores["Hyperglycemia"] += 25
        
        # Physical Inactivity component (from wearable data)
        if "daily_steps" in wearable_data:
            if wearable_data["daily_steps"] <= self.wearable_thresholds["daily_steps"]["high_risk"]:
                component_scores["Physical Inactivity"] += 50
            elif wearable_data["daily_steps"] <= self.wearable_thresholds["daily_steps"]["moderate_risk"]:
                component_scores["Physical Inactivity"] += 25
        
        if "active_minutes" in wearable_data:
            if wearable_data["active_minutes"] <= self.wearable_thresholds["active_minutes"]["high_risk"]:
                component_scores["Physical Inactivity"] += 50
            elif wearable_data["active_minutes"] <= self.wearable_thresholds["active_minutes"]["moderate_risk"]:
                component_scores["Physical Inactivity"] += 25
        
        # Normalize component scores to 0-100
        for component in component_scores:
            component_scores[component] = min(component_scores[component], 100)
        
        # Create risk report
        risk_report = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "clinical_score": clinical_score,
            "wearable_score": wearable_score,
            "combined_score": combined_score,
            "risk_category": risk_category,
            "metabolic_syndrome_subtype": subtype,
            "subtype_description": self.metabolic_syndrome_subtypes[subtype]["description"],
            "component_scores": component_scores,
            "recommendations": recommendations,
            "clinical_data": clinical_data,
            "wearable_data": wearable_data,
            "gender": gender
        }
        
        return risk_report
    
    def save_risk_report(self, risk_report: Dict[str, Any], file_path: Optional[str] = None) -> str:
        """
        Save risk report to a file.
        
        Args:
            risk_report: Risk report dictionary
            file_path: Path to save the risk report (default: OUTPUT_DIR/risk_report_{timestamp}.json)
        
        Returns:
            Path to the saved file
        """
        if file_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(OUTPUT_DIR, f"risk_report_{timestamp}.json")
        
        try:
            with open(file_path, "w") as f:
                json.dump(risk_report, f, indent=2)
            
            logger.info(f"Successfully saved risk report to {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"Error saving risk report: {str(e)}")
            return ""
    
    def visualize_risk_radar(self, component_scores: Dict[str, float]) -> plt.Figure:
        """
        Create a radar chart visualization of component scores.
        
        Args:
            component_scores: Dictionary of component scores
        
        Returns:
            Matplotlib figure
        """
        # Set up the radar chart
        categories = list(component_scores.keys())
        values = [component_scores[category] for category in categories]
        
        # Add the first value at the end to close the polygon
        values += [values[0]]
        categories += [categories[0]]
        
        # Calculate angles for each category
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += [angles[0]]
        
        # Create figure and polar axis
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Plot data
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        # Set category labels
        ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
        
        # Set radial limits
        ax.set_ylim(0, 100)
        
        # Add grid lines and labels
        ax.set_rgrids([25, 50, 75, 100], ['25', '50', '75', '100'])
        ax.grid(True)
        
        # Set title
        ax.set_title('Metabolic Syndrome Component Risk Scores', size=15, pad=20)
        
        return fig
    
    def visualize_risk_gauge(self, risk_score: float, risk_category: str) -> plt.Figure:
        """
        Create a gauge chart visualization of the risk score.
        
        Args:
            risk_score: Risk score (0-100)
            risk_category: Risk category
        
        Returns:
            Matplotlib figure
        """
        # Define gauge chart properties
        gauge_min = 0
        gauge_max = 100
        
        # Define colors for different risk categories
        category_colors = {
            "low": "green",
            "moderate": "yellow",
            "high": "orange",
            "very_high": "red"
        }
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Hide axis
        ax.axis('off')
        
        # Draw gauge background
        ax.add_patch(plt.Rectangle((0.1, 0.05), 0.8, 0.15, fill=True, color='lightgray'))
        
        # Draw gauge value
        gauge_width = 0.8 * (risk_score / gauge_max)
        ax.add_patch(plt.Rectangle((0.1, 0.05), gauge_width, 0.15, fill=True, color=category_colors[risk_category]))
        
        # Add risk score text
        ax.text(0.5, 0.35, f"Risk Score: {risk_score:.1f}", ha='center', va='center', fontsize=15)
        
        # Add risk category text
        ax.text(0.5, 0.25, f"Risk Category: {risk_category.title()}", ha='center', va='center', fontsize=12)
        
        # Add gauge labels
        ax.text(0.1, 0.02, '0', ha='center', va='center')
        ax.text(0.3, 0.02, '25', ha='center', va='center')
        ax.text(0.5, 0.02, '50', ha='center', va='center')
        ax.text(0.7, 0.02, '75', ha='center', va='center')
        ax.text(0.9, 0.02, '100', ha='center', va='center')
        
        # Add category labels
        ax.text(0.2, 0.0, 'Low', ha='center', va='center', color='green')
        ax.text(0.4, 0.0, 'Moderate', ha='center', va='center', color='yellow')
        ax.text(0.6, 0.0, 'High', ha='center', va='center', color='orange')
        ax.text(0.8, 0.0, 'Very High', ha='center', va='center', color='red')
        
        return fig
    
    def visualize_feature_importance(self) -> plt.Figure:
        """
        Create a bar chart visualization of feature importances.
        
        Returns:
            Matplotlib figure
        """
        # Sort feature importances
        sorted_importances = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
        features = [item[0] for item in sorted_importances]
        importances = [item[1] for item in sorted_importances]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bar chart
        bars = ax.barh(features, importances)
        
        # Color bars based on feature type
        for i, bar in enumerate(bars):
            if features[i] in ["waist_circumference", "triglycerides", "hdl_cholesterol", "systolic_bp", "diastolic_bp", "fasting_glucose"]:
                bar.set_color('skyblue')
            else:
                bar.set_color('lightgreen')
        
        # Add labels and title
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance for Metabolic Syndrome Risk Prediction')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='skyblue', label='Clinical Features'),
            Patch(facecolor='lightgreen', label='Wearable Features')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def visualize_wearable_trends(self, wearable_history: Dict[str, Dict[str, float]]) -> Dict[str, plt.Figure]:
        """
        Create line chart visualizations of wearable data trends.
        
        Args:
            wearable_history: Dictionary of wearable data history, where keys are dates and values are dictionaries of measurements
        
        Returns:
            Dictionary of Matplotlib figures
        """
        # Convert wearable history to DataFrame
        df = pd.DataFrame.from_dict(wearable_history, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Create figures for each wearable measurement
        figures = {}
        
        # Define measurements to visualize
        measurements = [
            ("daily_steps", "Daily Steps", "steps"),
            ("active_minutes", "Active Minutes", "minutes"),
            ("sleep_duration", "Sleep Duration", "hours"),
            ("resting_heart_rate", "Resting Heart Rate", "bpm"),
            ("heart_rate_variability", "Heart Rate Variability", "ms"),
            ("glucose_variability", "Glucose Variability", "%"),
            ("time_in_range", "Time in Range", "%"),
            ("stress_score", "Stress Score", "score")
        ]
        
        for measurement, title, unit in measurements:
            if measurement in df.columns:
                # Create figure and axis
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot line chart
                ax.plot(df.index, df[measurement], marker='o', linestyle='-', markersize=4)
                
                # Add reference lines for thresholds if available
                if measurement in self.wearable_thresholds:
                    thresholds = self.wearable_thresholds[measurement]
                    
                    # For measurements where higher is better
                    if measurement in ["daily_steps", "active_minutes", "sleep_duration", "heart_rate_variability", "time_in_range"]:
                        if "high_risk" in thresholds:
                            ax.axhline(y=thresholds["high_risk"], color='red', linestyle='--', alpha=0.7, label=f'High Risk (<{thresholds["high_risk"]})')
                        if "moderate_risk" in thresholds:
                            ax.axhline(y=thresholds["moderate_risk"], color='orange', linestyle='--', alpha=0.7, label=f'Moderate Risk (<{thresholds["moderate_risk"]})')
                        if "low_risk" in thresholds:
                            ax.axhline(y=thresholds["low_risk"], color='green', linestyle='--', alpha=0.7, label=f'Low Risk (≥{thresholds["low_risk"]})')
                    # For measurements where lower is better
                    else:
                        if "high_risk" in thresholds:
                            ax.axhline(y=thresholds["high_risk"], color='red', linestyle='--', alpha=0.7, label=f'High Risk (≥{thresholds["high_risk"]})')
                        if "moderate_risk" in thresholds:
                            ax.axhline(y=thresholds["moderate_risk"], color='orange', linestyle='--', alpha=0.7, label=f'Moderate Risk (≥{thresholds["moderate_risk"]})')
                        if "low_risk" in thresholds:
                            ax.axhline(y=thresholds["low_risk"], color='green', linestyle='--', alpha=0.7, label=f'Low Risk (<{thresholds["low_risk"]})')
                
                # Add labels and title
                ax.set_xlabel('Date')
                ax.set_ylabel(f'{title} ({unit})')
                ax.set_title(f'{title} Trend')
                
                # Add legend if thresholds are available
                if measurement in self.wearable_thresholds:
                    ax.legend()
                
                # Format x-axis dates
                fig.autofmt_xdate()
                
                # Adjust layout
                plt.tight_layout()
                
                # Add figure to dictionary
                figures[measurement] = fig
        
        return figures
    
    def visualize_risk_trend(self, risk_history: Dict[str, float]) -> plt.Figure:
        """
        Create a line chart visualization of risk score trend.
        
        Args:
            risk_history: Dictionary of risk scores, where keys are dates and values are risk scores
        
        Returns:
            Matplotlib figure
        """
        # Convert risk history to DataFrame
        df = pd.DataFrame.from_dict(risk_history, orient='index', columns=['risk_score'])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot line chart
        ax.plot(df.index, df['risk_score'], marker='o', linestyle='-', color='purple', markersize=4)
        
        # Add reference lines for risk categories
        ax.axhspan(0, 25, alpha=0.2, color='green', label='Low Risk')
        ax.axhspan(25, 50, alpha=0.2, color='yellow', label='Moderate Risk')
        ax.axhspan(50, 75, alpha=0.2, color='orange', label='High Risk')
        ax.axhspan(75, 100, alpha=0.2, color='red', label='Very High Risk')
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Risk Score')
        ax.set_title('Metabolic Syndrome Risk Score Trend')
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        return fig

def create_streamlit_app():
    """Create a Streamlit app for the Metabolic Syndrome Risk Calculator."""
    
    # Set page config
    st.set_page_config(
        page_title="Metabolic Syndrome Risk Calculator",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize risk calculator
    risk_calculator = MetabolicSyndromeRiskCalculator()
    
    # Add title and description
    st.title("Metabolic Syndrome Risk Calculator")
    st.markdown("""
    This interactive tool calculates your risk of metabolic syndrome based on clinical measurements and wearable device data.
    Metabolic syndrome is a cluster of conditions that occur together, increasing your risk of heart disease, stroke, and type 2 diabetes.
    """)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Risk Calculator", "Wearable Data Trends", "Knowledge Graph", "About"])
    
    # Tab 1: Risk Calculator
    with tab1:
        st.header("Calculate Your Metabolic Syndrome Risk")
        
        # Create columns for input sections
        col1, col2 = st.columns(2)
        
        # Column 1: Clinical Data
        with col1:
            st.subheader("Clinical Data")
            
            # Personal information
            gender = st.radio("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=18, max_value=100, value=45)
            
            # Clinical measurements
            st.markdown("#### Body Measurements")
            waist_circumference = st.number_input("Waist Circumference (cm)", min_value=50.0, max_value=200.0, value=95.0, step=0.1)
            bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=50.0, value=27.5, step=0.1)
            
            st.markdown("#### Blood Pressure")
            systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=220, value=125)
            diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=140, value=82)
            
            st.markdown("#### Blood Tests")
            fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=70.0, max_value=300.0, value=98.0, step=0.1)
            hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=14.0, value=5.6, step=0.1)
            triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=50.0, max_value=500.0, value=145.0, step=0.1)
            hdl_cholesterol = st.number_input("HDL Cholesterol (mg/dL)", min_value=20.0, max_value=100.0, value=45.0, step=0.1)
            ldl_cholesterol = st.number_input("LDL Cholesterol (mg/dL)", min_value=50.0, max_value=300.0, value=120.0, step=0.1)
        
        # Column 2: Wearable Data
        with col2:
            st.subheader("Wearable Device Data")
            
            # Physical activity
            st.markdown("#### Physical Activity")
            daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=7500)
            active_minutes = st.number_input("Active Minutes per Day", min_value=0, max_value=300, value=25)
            
            # Sleep
            st.markdown("#### Sleep")
            sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=12.0, value=7.0, step=0.1)
            
            # Heart rate
            st.markdown("#### Heart Rate")
            resting_heart_rate = st.number_input("Resting Heart Rate (bpm)", min_value=40, max_value=120, value=68)
            heart_rate_variability = st.number_input("Heart Rate Variability (ms)", min_value=0, max_value=100, value=35)
            
            # Glucose
            st.markdown("#### Glucose")
            glucose_variability = st.number_input("Glucose Variability (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
            time_in_range = st.number_input("Time in Range (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
            
            # Stress
            st.markdown("#### Stress")
            stress_score = st.number_input("Stress Score (0-100)", min_value=0, max_value=100, value=40)
        
        # Create dictionaries for clinical and wearable data
        clinical_data = {
            "waist_circumference": waist_circumference,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "fasting_glucose": fasting_glucose,
            "hba1c": hba1c,
            "triglycerides": triglycerides,
            "hdl_cholesterol": hdl_cholesterol,
            "ldl_cholesterol": ldl_cholesterol
        }
        
        wearable_data = {
            "daily_steps": daily_steps,
            "active_minutes": active_minutes,
            "sleep_duration": sleep_duration,
            "resting_heart_rate": resting_heart_rate,
            "heart_rate_variability": heart_rate_variability,
            "glucose_variability": glucose_variability,
            "time_in_range": time_in_range,
            "stress_score": stress_score
        }
        
        # Calculate button
        if st.button("Calculate Risk"):
            # Generate risk report
            risk_report = risk_calculator.generate_risk_report(clinical_data, wearable_data, gender)
            
            # Display results
            st.markdown("---")
            st.header("Risk Assessment Results")
            
            # Create columns for results
            res_col1, res_col2 = st.columns(2)
            
            # Column 1: Risk scores and category
            with res_col1:
                st.subheader("Risk Scores")
                
                # Create gauge chart for combined risk score
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_report["combined_score"],
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Combined Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 25], "color": "green"},
                            {"range": [25, 50], "color": "yellow"},
                            {"range": [50, 75], "color": "orange"},
                            {"range": [75, 100], "color": "red"}
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": risk_report["combined_score"]
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Display risk category
                st.markdown(f"**Risk Category:** {risk_report['risk_category'].title()}")
                
                # Display individual scores
                st.markdown(f"**Clinical Score:** {risk_report['clinical_score']:.1f}")
                st.markdown(f"**Wearable Score:** {risk_report['wearable_score']:.1f}")
                
                # Display metabolic syndrome subtype
                st.subheader("Metabolic Syndrome Subtype")
                st.markdown(f"**Subtype:** {risk_report['metabolic_syndrome_subtype']}")
                st.markdown(f"**Description:** {risk_report['subtype_description']}")
            
            # Column 2: Component scores radar chart
            with res_col2:
                st.subheader("Component Risk Scores")
                
                # Create radar chart for component scores
                categories = list(risk_report["component_scores"].keys())
                values = list(risk_report["component_scores"].values())
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Component Scores'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=False
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Display component scores
                for component, score in risk_report["component_scores"].items():
                    st.markdown(f"**{component}:** {score:.1f}")
            
            # Recommendations
            st.subheader("Personalized Recommendations")
            for i, recommendation in enumerate(risk_report["recommendations"], 1):
                st.markdown(f"{i}. {recommendation}")
            
            # Save report button
            if st.button("Save Risk Report"):
                # Save risk report to file
                report_path = risk_calculator.save_risk_report(risk_report)
                
                if report_path:
                    st.success(f"Risk report saved to {report_path}")
                else:
                    st.error("Failed to save risk report")
    
    # Tab 2: Wearable Data Trends
    with tab2:
        st.header("Wearable Data Trends")
        st.markdown("""
        This section allows you to visualize trends in your wearable device data over time.
        Upload a CSV file with your wearable data history or use the sample data.
        """)
        
        # Option to use sample data or upload file
        data_option = st.radio("Data Source", ["Use Sample Data", "Upload CSV File"])
        
        wearable_history = {}
        
        if data_option == "Use Sample Data":
            # Generate sample wearable data for the past 30 days
            np.random.seed(42)
            
            for i in range(30):
                date = (datetime.datetime.now() - datetime.timedelta(days=30-i)).strftime("%Y-%m-%d")
                
                # Generate improving trend
                daily_steps = 6000 + i * 100 + np.random.normal(0, 500)
                active_minutes = 20 + i * 0.5 + np.random.normal(0, 5)
                sleep_duration = 6.5 + i * 0.05 + np.random.normal(0, 0.3)
                resting_heart_rate = 75 - i * 0.2 + np.random.normal(0, 3)
                heart_rate_variability = 25 + i * 0.5 + np.random.normal(0, 3)
                
                # Calculate risk score (decreasing over time)
                risk_score = 70 - i * 1.5 + np.random.normal(0, 5)
                risk_score = max(0, min(100, risk_score))
                
                # Store data for this day
                wearable_history[date] = {
                    "daily_steps": daily_steps,
                    "active_minutes": active_minutes,
                    "sleep_duration": sleep_duration,
                    "resting_heart_rate": resting_heart_rate,
                    "heart_rate_variability": heart_rate_variability,
                    "risk_score": risk_score
                }
        
        else:  # Upload CSV File
            uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    # Read CSV file
                    df = pd.read_csv(uploaded_file)
                    
                    # Check if required columns exist
                    required_columns = ["date"]
                    if not all(col in df.columns for col in required_columns):
                        st.error("CSV file must contain a 'date' column")
                    else:
                        # Convert DataFrame to dictionary
                        df.set_index("date", inplace=True)
                        wearable_history = df.to_dict(orient="index")
                
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
        
        # Display wearable data trends
        if wearable_history:
            # Convert wearable history to DataFrame
            df = pd.DataFrame.from_dict(wearable_history, orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Display data table
            st.subheader("Wearable Data History")
            st.dataframe(df)
            
            # Create tabs for different visualizations
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Physical Activity", "Physiological Metrics", "Risk Score"])
            
            # Tab 1: Physical Activity
            with viz_tab1:
                st.subheader("Physical Activity Trends")
                
                # Daily Steps
                if "daily_steps" in df.columns:
                    fig_steps = px.line(df, y="daily_steps", title="Daily Steps Trend")
                    fig_steps.add_hline(y=10000, line_dash="dash", line_color="green", annotation_text="Target")
                    fig_steps.add_hline(y=7000, line_dash="dash", line_color="orange")
                    fig_steps.add_hline(y=5000, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_steps, use_container_width=True)
                
                # Active Minutes
                if "active_minutes" in df.columns:
                    fig_active = px.line(df, y="active_minutes", title="Active Minutes Trend")
                    fig_active.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Target")
                    fig_active.add_hline(y=20, line_dash="dash", line_color="orange")
                    fig_active.add_hline(y=10, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_active, use_container_width=True)
            
            # Tab 2: Physiological Metrics
            with viz_tab2:
                st.subheader("Physiological Metrics Trends")
                
                # Sleep Duration
                if "sleep_duration" in df.columns:
                    fig_sleep = px.line(df, y="sleep_duration", title="Sleep Duration Trend")
                    fig_sleep.add_hline(y=8, line_dash="dash", line_color="green", annotation_text="Target")
                    fig_sleep.add_hline(y=7, line_dash="dash", line_color="orange")
                    fig_sleep.add_hline(y=6, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_sleep, use_container_width=True)
                
                # Resting Heart Rate
                if "resting_heart_rate" in df.columns:
                    fig_hr = px.line(df, y="resting_heart_rate", title="Resting Heart Rate Trend")
                    fig_hr.add_hline(y=60, line_dash="dash", line_color="green", annotation_text="Target")
                    fig_hr.add_hline(y=70, line_dash="dash", line_color="orange")
                    fig_hr.add_hline(y=80, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_hr, use_container_width=True)
                
                # Heart Rate Variability
                if "heart_rate_variability" in df.columns:
                    fig_hrv = px.line(df, y="heart_rate_variability", title="Heart Rate Variability Trend")
                    fig_hrv.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Target")
                    fig_hrv.add_hline(y=30, line_dash="dash", line_color="orange")
                    fig_hrv.add_hline(y=20, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_hrv, use_container_width=True)
            
            # Tab 3: Risk Score
            with viz_tab3:
                st.subheader("Risk Score Trend")
                
                # Risk Score
                if "risk_score" in df.columns:
                    fig_risk = px.line(df, y="risk_score", title="Metabolic Syndrome Risk Score Trend")
                    fig_risk.add_hrect(y0=0, y1=25, line_width=0, fillcolor="green", opacity=0.2)
                    fig_risk.add_hrect(y0=25, y1=50, line_width=0, fillcolor="yellow", opacity=0.2)
                    fig_risk.add_hrect(y0=50, y1=75, line_width=0, fillcolor="orange", opacity=0.2)
                    fig_risk.add_hrect(y0=75, y1=100, line_width=0, fillcolor="red", opacity=0.2)
                    fig_risk.update_layout(yaxis_range=[0, 100])
                    st.plotly_chart(fig_risk, use_container_width=True)
                    
                    # Calculate risk reduction
                    if len(df) > 1:
                        first_risk = df["risk_score"].iloc[0]
                        last_risk = df["risk_score"].iloc[-1]
                        risk_change = last_risk - first_risk
                        
                        if risk_change < 0:
                            st.success(f"Your risk score has decreased by {abs(risk_change):.1f} points over this period!")
                        elif risk_change > 0:
                            st.warning(f"Your risk score has increased by {risk_change:.1f} points over this period.")
                        else:
                            st.info("Your risk score has remained stable over this period.")
    
    # Tab 3: Knowledge Graph
    with tab3:
        st.header("Metabolic Syndrome Knowledge Graph")
        st.markdown("""
        This section visualizes the relationships between different components of metabolic syndrome,
        wearable measurements, and risk factors using a knowledge graph.
        """)
        
        # Create tabs for different visualizations
        kg_tab1, kg_tab2, kg_tab3 = st.tabs(["Overview", "Wearable Integration", "Risk Factors"])
        
        # Tab 1: Overview
        with kg_tab1:
            st.subheader("Metabolic Syndrome Components")
            
            # Display metabolic syndrome components diagram
            components_html = """
            <div style="text-align: center;">
                <svg width="800" height="500" xmlns="http://www.w3.org/2000/svg">
                    <!-- Central Node -->
                    <circle cx="400" cy="250" r="80" fill="#ff7f0e" stroke="black" stroke-width="2"/>
                    <text x="400" y="250" text-anchor="middle" alignment-baseline="middle" font-weight="bold">Metabolic Syndrome</text>
                    
                    <!-- Component Nodes -->
                    <circle cx="200" cy="150" r="60" fill="#1f77b4" stroke="black" stroke-width="2"/>
                    <text x="200" y="150" text-anchor="middle" alignment-baseline="middle" font-weight="bold">Obesity</text>
                    
                    <circle cx="600" cy="150" r="60" fill="#1f77b4" stroke="black" stroke-width="2"/>
                    <text x="600" y="150" text-anchor="middle" alignment-baseline="middle" font-weight="bold">Dyslipidemia</text>
                    
                    <circle cx="200" cy="350" r="60" fill="#1f77b4" stroke="black" stroke-width="2"/>
                    <text x="200" y="350" text-anchor="middle" alignment-baseline="middle" font-weight="bold">Hypertension</text>
                    
                    <circle cx="600" cy="350" r="60" fill="#1f77b4" stroke="black" stroke-width="2"/>
                    <text x="600" y="350" text-anchor="middle" alignment-baseline="middle" font-weight="bold">Hyperglycemia</text>
                    
                    <circle cx="400" cy="100" r="60" fill="#1f77b4" stroke="black" stroke-width="2"/>
                    <text x="400" y="100" text-anchor="middle" alignment-baseline="middle" font-weight="bold">Inflammation</text>
                    
                    <!-- Connecting Lines -->
                    <line x1="270" y1="150" x2="330" y2="210" stroke="black" stroke-width="2"/>
                    <line x1="530" y1="150" x2="470" y2="210" stroke="black" stroke-width="2"/>
                    <line x1="270" y1="350" x2="330" y2="290" stroke="black" stroke-width="2"/>
                    <line x1="530" y1="350" x2="470" y2="290" stroke="black" stroke-width="2"/>
                    <line x1="400" y1="160" x2="400" y2="180" stroke="black" stroke-width="2"/>
                    
                    <!-- Bidirectional Arrows -->
                    <line x1="250" y1="180" x2="550" y2="180" stroke="black" stroke-width="1" stroke-dasharray="5,5"/>
                    <line x1="250" y1="320" x2="550" y2="320" stroke="black" stroke-width="1" stroke-dasharray="5,5"/>
                    <line x1="230" y1="200" x2="230" y2="300" stroke="black" stroke-width="1" stroke-dasharray="5,5"/>
                    <line x1="570" y1="200" x2="570" y2="300" stroke="black" stroke-width="1" stroke-dasharray="5,5"/>
                </svg>
            </div>
            """
            
            st.markdown(components_html, unsafe_allow_html=True)
            
            # Display component descriptions
            st.markdown("""
            ### Metabolic Syndrome Components:
            
            1. **Obesity**: Central obesity with increased waist circumference
               - Male: ≥ 102 cm (40 inches)
               - Female: ≥ 88 cm (35 inches)
            
            2. **Dyslipidemia**: Abnormal blood lipid levels
               - Elevated triglycerides: ≥ 150 mg/dL
               - Reduced HDL cholesterol:
                 - Male: < 40 mg/dL
                 - Female: < 50 mg/dL
            
            3. **Hypertension**: Elevated blood pressure
               - Systolic: ≥ 130 mmHg
               - Diastolic: ≥ 85 mmHg
            
            4. **Hyperglycemia**: Elevated blood glucose
               - Fasting glucose: ≥ 100 mg/dL
               - HbA1c: ≥ 5.7%
            
            5. **Inflammation**: Chronic low-grade inflammation
               - Elevated inflammatory markers (CRP, IL-6, TNF-α)
            
            Metabolic syndrome is diagnosed when at least 3 of these 5 components are present.
            """)
        
        # Tab 2: Wearable Integration
        with kg_tab2:
            st.subheader("Wearable Data Integration")
            
            # Display wearable integration diagram
            wearable_html = """
            <div style="text-align: center;">
                <svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
                    <!-- Central Node -->
                    <circle cx="400" cy="300" r="80" fill="#ff7f0e" stroke="black" stroke-width="2"/>
                    <text x="400" y="300" text-anchor="middle" alignment-baseline="middle" font-weight="bold">Metabolic Syndrome</text>
                    
                    <!-- Component Nodes -->
                    <circle cx="200" cy="200" r="60" fill="#1f77b4" stroke="black" stroke-width="2"/>
                    <text x="200" y="200" text-anchor="middle" alignment-baseline="middle" font-weight="bold">Obesity</text>
                    
                    <circle cx="600" cy="200" r="60" fill="#1f77b4" stroke="black" stroke-width="2"/>
                    <text x="600" y="200" text-anchor="middle" alignment-baseline="middle" font-weight="bold">Dyslipidemia</text>
                    
                    <circle cx="200" cy="400" r="60" fill="#1f77b4" stroke="black" stroke-width="2"/>
                    <text x="200" y="400" text-anchor="middle" alignment-baseline="middle" font-weight="bold">Hypertension</text>
                    
                    <circle cx="600" cy="400" r="60" fill="#1f77b4" stroke="black" stroke-width="2"/>
                    <text x="600" y="400" text-anchor="middle" alignment-baseline="middle" font-weight="bold">Hyperglycemia</text>
                    
                    <!-- Wearable Nodes -->
                    <rect x="50" y="100" width="120" height="40" rx="10" fill="#2ca02c" stroke="black" stroke-width="2"/>
                    <text x="110" y="120" text-anchor="middle" alignment-baseline="middle">Daily Steps</text>
                    
                    <rect x="50" y="150" width="120" height="40" rx="10" fill="#2ca02c" stroke="black" stroke-width="2"/>
                    <text x="110" y="170" text-anchor="middle" alignment-baseline="middle">Active Minutes</text>
                    
                    <rect x="50" y="300" width="120" height="40" rx="10" fill="#2ca02c" stroke="black" stroke-width="2"/>
                    <text x="110" y="320" text-anchor="middle" alignment-baseline="middle">Resting HR</text>
                    
                    <rect x="50" y="350" width="120" height="40" rx="10" fill="#2ca02c" stroke="black" stroke-width="2"/>
                    <text x="110" y="370" text-anchor="middle" alignment-baseline="middle">HR Variability</text>
                    
                    <rect x="630" y="300" width="120" height="40" rx="10" fill="#2ca02c" stroke="black" stroke-width="2"/>
                    <text x="690" y="320" text-anchor="middle" alignment-baseline="middle">Glucose Var.</text>
                    
                    <rect x="630" y="350" width="120" height="40" rx="10" fill="#2ca02c" stroke="black" stroke-width="2"/>
                    <text x="690" y="370" text-anchor="middle" alignment-baseline="middle">Time in Range</text>
                    
                    <rect x="340" y="500" width="120" height="40" rx="10" fill="#2ca02c" stroke="black" stroke-width="2"/>
                    <text x="400" y="520" text-anchor="middle" alignment-baseline="middle">Sleep Duration</text>
                    
                    <rect x="340" y="550" width="120" height="40" rx="10" fill="#2ca02c" stroke="black" stroke-width="2"/>
                    <text x="400" y="570" text-anchor="middle" alignment-baseline="middle">Stress Score</text>
                    
                    <!-- Connecting Lines -->
                    <line x1="270" y1="200" x2="330" y2="260" stroke="black" stroke-width="2"/>
                    <line x1="530" y1="200" x2="470" y2="260" stroke="black" stroke-width="2"/>
                    <line x1="270" y1="400" x2="330" y2="340" stroke="black" stroke-width="2"/>
                    <line x1="530" y1="400" x2="470" y2="340" stroke="black" stroke-width="2"/>
                    
                    <!-- Wearable Connections -->
                    <line x1="170" y1="120" x2="200" y2="150" stroke="#2ca02c" stroke-width="2"/>
                    <line x1="170" y1="170" x2="200" y2="180" stroke="#2ca02c" stroke-width="2"/>
                    
                    <line x1="170" y1="320" x2="200" y2="380" stroke="#2ca02c" stroke-width="2"/>
                    <line x1="170" y1="370" x2="200" y2="390" stroke="#2ca02c" stroke-width="2"/>
                    
                    <line x1="630" y1="320" x2="600" y2="380" stroke="#2ca02c" stroke-width="2"/>
                    <line x1="630" y1="370" x2="600" y2="390" stroke="#2ca02c" stroke-width="2"/>
                    
                    <line x1="400" y1="500" x2="400" y2="380" stroke="#2ca02c" stroke-width="2"/>
                    <line x1="400" y1="550" x2="400" y2="380" stroke="#2ca02c" stroke-width="2"/>
                </svg>
            </div>
            """
            
            st.markdown(wearable_html, unsafe_allow_html=True)
            
            # Display wearable integration description
            st.markdown("""
            ### Wearable Data Integration with Metabolic Syndrome Components:
            
            1. **Physical Activity Metrics**
               - Daily Steps: Inversely associated with obesity and insulin resistance
               - Active Minutes: Improves insulin sensitivity and reduces cardiovascular risk
            
            2. **Heart Rate Metrics**
               - Resting Heart Rate: Elevated in hypertension and autonomic dysfunction
               - Heart Rate Variability: Reduced in autonomic dysfunction and stress
            
            3. **Glucose Metrics**
               - Glucose Variability: Indicates glycemic instability and insulin resistance
               - Time in Range: Reflects overall glycemic control
            
            4. **Sleep and Stress Metrics**
               - Sleep Duration: Poor sleep associated with insulin resistance and inflammation
               - Stress Score: Chronic stress contributes to metabolic dysfunction
            
            Wearable data provides continuous monitoring of physiological parameters that can detect early signs of metabolic syndrome before clinical manifestations appear.
            """)
        
        # Tab 3: Risk Factors
        with kg_tab3:
            st.subheader("Risk Factors and Subtypes")
            
            # Display risk factors and subtypes
            st.markdown("""
            ### Metabolic Syndrome Subtypes:
            
            1. **Obesity-Dominant**
               - Primary Feature: Central obesity
               - Clinical Characteristics: Increased waist circumference, elevated BMI, insulin resistance
               - Wearable Indicators: Low daily steps, low active minutes, high resting heart rate
               - Treatment Focus: Weight management, increased physical activity
            
            2. **Dyslipidemia-Dominant**
               - Primary Feature: Abnormal lipid levels
               - Clinical Characteristics: Elevated triglycerides, reduced HDL cholesterol, elevated LDL cholesterol
               - Wearable Indicators: Variable daily steps, variable active minutes, moderate glucose variability
               - Treatment Focus: Dietary fat modification, omega-3 supplementation, aerobic exercise
            
            3. **Hypertension-Dominant**
               - Primary Feature: Elevated blood pressure
               - Clinical Characteristics: Elevated systolic and diastolic blood pressure, elevated pulse pressure
               - Wearable Indicators: High resting heart rate, low heart rate variability, high stress score
               - Treatment Focus: Sodium restriction, DASH diet, stress reduction techniques
            
            4. **Hyperglycemia-Dominant**
               - Primary Feature: Elevated glucose levels
               - Clinical Characteristics: Elevated fasting glucose, elevated HbA1c, insulin resistance
               - Wearable Indicators: High glucose variability, low time in range, high resting heart rate
               - Treatment Focus: Carbohydrate modification, post-meal exercise, continuous glucose monitoring
            
            5. **Mixed/Balanced**
               - Primary Feature: Multiple components with similar severity
               - Clinical Characteristics: Moderate abnormalities across multiple parameters
               - Wearable Indicators: Moderate daily steps, moderate active minutes, moderate heart rate variability
               - Treatment Focus: Comprehensive lifestyle modification, Mediterranean diet, regular monitoring
            """)
            
            # Display risk calculator model information
            st.subheader("Risk Calculator Model")
            
            # Feature importance chart
            feature_importances = risk_calculator.feature_importances
            sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            features = [item[0] for item in sorted_importances]
            importances = [item[1] for item in sorted_importances]
            
            fig_importance = px.bar(
                x=importances,
                y=features,
                orientation='h',
                title='Feature Importance for Metabolic Syndrome Risk Prediction',
                labels={'x': 'Importance', 'y': 'Feature'}
            )
            
            # Color bars based on feature type
            fig_importance.update_traces(
                marker_color=[
                    'skyblue' if feature in ["waist_circumference", "triglycerides", "hdl_cholesterol", "systolic_bp", "diastolic_bp", "fasting_glucose"]
                    else 'lightgreen' for feature in features
                ]
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Model information
            st.markdown(f"""
            **Model Type:** {risk_calculator.model_type}
            
            **Risk Categories:**
            - Low Risk: 0-25
            - Moderate Risk: 25-50
            - High Risk: 50-75
            - Very High Risk: 75-100
            
            The risk calculator integrates both clinical measurements and wearable device data to provide a comprehensive assessment of metabolic syndrome risk. Clinical measurements are weighted more heavily (70%) than wearable data (30%) in the combined risk score calculation.
            """)
    
    # Tab 4: About
    with tab4:
        st.header("About Metabolic Syndrome")
        
        # What is Metabolic Syndrome
        st.subheader("What is Metabolic Syndrome?")
        st.markdown("""
        Metabolic syndrome is a cluster of conditions that occur together, increasing your risk of heart disease, stroke, and type 2 diabetes. These conditions include:
        
        - Increased blood pressure
        - High blood sugar
        - Excess body fat around the waist
        - Abnormal cholesterol or triglyceride levels
        
        Having just one of these conditions doesn't mean you have metabolic syndrome. However, any of these conditions increase your risk of serious disease. Having more than one of these conditions increases your risk even more.
        
        Metabolic syndrome is becoming increasingly common worldwide, largely due to increases in obesity rates. Lifestyle modifications are the primary treatment for metabolic syndrome.
        """)
        
        # Diagnostic Criteria
        st.subheader("Diagnostic Criteria")
        st.markdown("""
        According to the National Cholesterol Education Program Adult Treatment Panel III (NCEP ATP III), metabolic syndrome is diagnosed when a person has at least three of the following five conditions:
        
        1. **Abdominal obesity**: Waist circumference ≥ 102 cm (40 inches) in men or ≥ 88 cm (35 inches) in women
        2. **Elevated triglycerides**: ≥ 150 mg/dL (1.7 mmol/L) or on drug treatment for elevated triglycerides
        3. **Reduced HDL cholesterol**: < 40 mg/dL (1.03 mmol/L) in men or < 50 mg/dL (1.29 mmol/L) in women or on drug treatment for reduced HDL cholesterol
        4. **Elevated blood pressure**: ≥ 130 mm Hg systolic or ≥ 85 mm Hg diastolic or on antihypertensive drug treatment
        5. **Elevated fasting glucose**: ≥ 100 mg/dL (5.6 mmol/L) or on drug treatment for elevated glucose
        """)
        
        # Risk Factors
        st.subheader("Risk Factors")
        st.markdown("""
        The following factors increase your risk of developing metabolic syndrome:
        
        - **Age**: The risk of metabolic syndrome increases with age
        - **Ethnicity**: Hispanics and Asians appear to be at greater risk
        - **Obesity**: Having a body mass index (BMI) greater than 25
        - **Diabetes**: You're more likely to have metabolic syndrome if you have a family history of type 2 diabetes
        - **Other diseases**: Your risk of metabolic syndrome is higher if you've ever had cardiovascular disease, nonalcoholic fatty liver disease, or polycystic ovary syndrome
        """)
        
        # Complications
        st.subheader("Complications")
        st.markdown("""
        Having metabolic syndrome can increase your risk of developing:
        
        - **Type 2 diabetes**: If you don't make lifestyle changes to control your excess weight, you may develop insulin resistance, which can cause your blood sugar levels to rise
        - **Cardiovascular disease**: High cholesterol and high blood pressure can contribute to the buildup of plaques in your arteries, which can narrow and harden your arteries
        - **Nonalcoholic fatty liver disease**: This occurs when fat accumulates in the liver
        - **Sleep apnea**: This sleep disorder is characterized by repeated stopping and starting of breathing during sleep
        """)
        
        # Prevention and Treatment
        st.subheader("Prevention and Treatment")
        st.markdown("""
        The key to preventing or reversing metabolic syndrome is adopting healthy lifestyle changes:
        
        - **Regular physical activity**: Aim for at least 150 minutes of moderate-intensity aerobic activity per week
        - **Weight loss**: Losing 7-10% of your body weight can reduce insulin resistance and blood pressure
        - **Healthy diet**: Focus on fruits, vegetables, whole grains, lean proteins, and low-fat dairy
        - **Smoking cessation**: Quitting smoking can reduce your risk of heart disease
        - **Stress reduction**: Chronic stress contributes to metabolic syndrome
        - **Regular monitoring**: Regular check-ups to monitor blood pressure, cholesterol, and blood sugar
        
        In some cases, medication may be necessary to control specific risk factors, such as high blood pressure, high cholesterol, or high blood sugar.
        """)
        
        # About the Calculator
        st.subheader("About this Risk Calculator")
        st.markdown("""
        This Metabolic Syndrome Risk Calculator was developed as part of a research project on knowledge representation and ontologies for metabolic syndrome using the AllOfUS dataset. The calculator integrates:
        
        1. **Clinical measurements** from standard medical tests
        2. **Wearable device data** from fitness trackers, smartwatches, and continuous glucose monitors
        3. **Knowledge representation** using ontologies and knowledge graphs
        4. **Machine learning models** for risk prediction
        
        The calculator provides:
        
        - Comprehensive risk assessment for metabolic syndrome
        - Identification of metabolic syndrome subtypes
        - Personalized recommendations based on risk profile
        - Visualization of risk factors and trends
        
        This tool is intended for educational and research purposes only and should not replace professional medical advice. Always consult with a healthcare provider for diagnosis and treatment of medical conditions.
        """)
        
        # References
        st.subheader("References")
        st.markdown("""
        1. Alberti KG, Eckel RH, Grundy SM, et al. Harmonizing the metabolic syndrome: a joint interim statement of the International Diabetes Federation Task Force on Epidemiology and Prevention; National Heart, Lung, and Blood Institute; American Heart Association; World Heart Federation; International Atherosclerosis Society; and International Association for the Study of Obesity. Circulation. 2009;120(16):1640-1645.
        
        2. Grundy SM, Cleeman JI, Daniels SR, et al. Diagnosis and management of the metabolic syndrome: an American Heart Association/National Heart, Lung, and Blood Institute Scientific Statement. Circulation. 2005;112(17):2735-2752.
        
        3. Saklayen MG. The Global Epidemic of the Metabolic Syndrome. Curr Hypertens Rep. 2018;20(2):12.
        
        4. Sperling LS, Mechanick JI, Neeland IJ, et al. The CardioMetabolic Health Alliance: Working Toward a New Care Model for the Metabolic Syndrome. J Am Coll Cardiol. 2015;66(9):1050-1067.
        
        5. Ussery-Hall A, Kaur H, Whiteman MK, et al. Prevalence of selected risk behaviors and chronic diseases and conditions-steps communities, United States, 2006-2007. MMWR Surveill Summ. 2010;59(8):1-37.
        """)

def main():
    """Main function to run the Streamlit app."""
    create_streamlit_app()

if __name__ == "__main__":
    main()
