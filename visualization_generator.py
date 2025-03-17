"""
Visualization Generator for Metabolic Syndrome Research

This module provides functionality to generate visualizations for metabolic syndrome research
using clinical and wearable data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick
from datetime import datetime
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import json
from scipy import stats
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualization_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("visualization_generator")

class VisualizationGenerator:
    """
    Class for generating visualizations for metabolic syndrome research.
    """
    
    def __init__(self, config=None):
        """
        Initialize the visualization generator.
        
        Args:
            config: Configuration dictionary with visualization parameters
        """
        # Default configuration
        self.config = {
            'output_dir': 'figures',
            'figure_format': 'png',
            'figure_dpi': 300,
            'figure_width': 10,
            'figure_height': 6,
            'color_palette': 'viridis',
            'style': 'whitegrid',
            'context': 'notebook',
            'font_scale': 1.2
        }
        
        # Update with user-provided configuration
        if config:
            self.config.update(config)
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Set visualization style
        sns.set_style(self.config['style'])
        sns.set_context(self.config['context'], font_scale=self.config['font_scale'])
        
        # Initialize data containers
        self.data = {}
        self.visualizations = {}
        
        logger.info("VisualizationGenerator initialized with configuration:")
        for key, value in self.config.items():
            logger.info(f"  {key}: {value}")
    
    def set_data(self, data_dict):
        """
        Set data for visualization.
        
        Args:
            data_dict: Dictionary with data for visualization
            
        Returns:
            None
        """
        self.data = data_dict
        
        logger.info("Data set for visualization:")
        for key, value in self.data.items():
            if isinstance(value, pd.DataFrame):
                logger.info(f"  {key}: DataFrame with {len(value)} rows and {len(value.columns)} columns")
            else:
                logger.info(f"  {key}: {type(value)}")
    
    def save_figure(self, fig, filename, close_fig=True):
        """
        Save figure to file.
        
        Args:
            fig: Figure to save
            filename: Filename for the figure
            close_fig: Whether to close the figure after saving
            
        Returns:
            Path to saved figure
        """
        # Create full path
        if not filename.endswith(f".{self.config['figure_format']}"):
            filename = f"{filename}.{self.config['figure_format']}"
        
        file_path = os.path.join(self.config['output_dir'], filename)
        
        # Save figure
        fig.savefig(file_path, dpi=self.config['figure_dpi'], bbox_inches='tight')
        
        # Close figure if requested
        if close_fig:
            plt.close(fig)
        
        logger.info(f"Saved figure to {file_path}")
        
        return file_path
    
    def create_prevalence_visualizations(self):
        """
        Create visualizations for metabolic syndrome prevalence.
        
        Returns:
            Dictionary with paths to saved visualizations
        """
        logger.info("Creating metabolic syndrome prevalence visualizations")
        
        # Check if we have the necessary data
        if 'integrated_data' not in self.data and 'clinical_data' not in self.data:
            logger.error("No integrated or clinical data available for prevalence visualizations")
            return {}
        
        # Use integrated data if available, otherwise use clinical data
        data = self.data.get('integrated_data', self.data.get('clinical_data'))
        
        # Check if metabolic syndrome status is available
        if 'has_metabolic_syndrome' not in data.columns:
            logger.error("Metabolic syndrome status not available in data")
            return {}
        
        # Dictionary to store visualization paths
        prevalence_viz = {}
        
        # --- Visualization 1: Overall Prevalence ---
        
        # Calculate overall prevalence
        overall_prevalence = data['has_metabolic_syndrome'].mean() * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=(self.config['figure_width'], self.config['figure_height']))
        
        # Create bar chart
        ax.bar(['No MetS', 'MetS'], 
               [(100 - overall_prevalence), overall_prevalence],
               color=['#1f77b4', '#ff7f0e'])
        
        # Add labels
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Metabolic Syndrome Prevalence in Study Population', fontsize=14)
        
        # Add value labels
        ax.text(0, (100 - overall_prevalence) / 2, f"{100 - overall_prevalence:.1f}%", 
               ha='center', va='center', fontsize=12)
        ax.text(1, overall_prevalence / 2, f"{overall_prevalence:.1f}%", 
               ha='center', va='center', fontsize=12)
        
        # Add grid
        ax.grid(axis='y', alpha=0.3)
        
        # Save figure
        fig_path = self.save_figure(fig, "overall_prevalence")
        prevalence_viz['overall_prevalence'] = fig_path
        
        # --- Visualization 2: Prevalence by Demographics ---
        
        # Check if demographic variables are available
        demographic_vars = []
        for var in ['sex', 'age_group', 'race']:
            if var in data.columns:
                demographic_vars.append(var)
        
        if not demographic_vars:
            logger.warning("No demographic variables available for prevalence visualizations")
        else:
            # Create age groups if not already present
            if 'age' in data.columns and 'age_group' not in data.columns:
                data['age_group'] = pd.cut(
                    data['age'],
                    bins=[0, 30, 40, 50, 60, 70, 100],
                    labels=['<30', '30-39', '40-49', '50-59', '60-69', '70+']
                )
                demographic_vars.append('age_group')
            
            # Create figure with subplots
            n_plots = len(demographic_vars)
            fig, axes = plt.subplots(1, n_plots, figsize=(self.config['figure_width'] * n_plots, self.config['figure_height']))
            
            # Handle single subplot case
            if n_plots == 1:
                axes = [axes]
            
            # Create prevalence plots for each demographic variable
            for i, var in enumerate(demographic_vars):
                # Calculate prevalence by group
                prevalence_by_group = data.groupby(var)['has_metabolic_syndrome'].mean() * 100
                
                # Sort by prevalence for better visualization
                prevalence_by_group = prevalence_by_group.sort_values(ascending=False)
                
                # Create bar chart
                prevalence_by_group.plot(kind='bar', ax=axes[i], color=sns.color_palette(self.config['color_palette'], len(prevalence_by_group)))
                
                # Add labels
                axes[i].set_ylabel('Prevalence (%)')
                axes[i].set_title(f'Prevalence by {var.replace("_", " ").title()}', fontsize=12)
                
                # Add value labels
                for j, v in enumerate(prevalence_by_group):
                    axes[i].text(j, v + 1, f"{v:.1f}%", ha='center', fontsize=10)
                
                # Add grid
                axes[i].grid(axis='y', alpha=0.3)
                
                # Rotate x-axis labels for better readability
                axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            fig_path = self.save_figure(fig, "prevalence_by_demographics")
            prevalence_viz['prevalence_by_demographics'] = fig_path
        
        # --- Visualization 3: Criteria Distribution ---
        
        # Check if criteria columns are available
        criteria_columns = [col for col in data.columns if col.startswith('meets_') and col.endswith('_criterion')]
        
        if not criteria_columns:
            logger.warning("No criteria columns available for distribution visualization")
        else:
            # Create figure
            fig, ax = plt.subplots(figsize=(self.config['figure_width'], self.config['figure_height']))
            
            # Calculate prevalence for each criterion
            criteria_prevalence = {}
            for col in criteria_columns:
                name = col.replace('meets_', '').replace('_criterion', '').replace('_', ' ').title()
                criteria_prevalence[name] = data[col].mean() * 100
            
            # Convert to Series and sort
            criteria_series = pd.Series(criteria_prevalence).sort_values(ascending=False)
            
            # Create bar chart
            criteria_series.plot(kind='bar', ax=ax, color=sns.color_palette(self.config['color_palette'], len(criteria_series)))
            
            # Add labels
            ax.set_ylabel('Prevalence (%)')
            ax.set_title('Prevalence of Individual Metabolic Syndrome Criteria', fontsize=14)
            
            # Add value labels
            for i, v in enumerate(criteria_series):
                ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=10)
            
            # Add grid
            ax.grid(axis='y', alpha=0.3)
            
            # Rotate x-axis labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Save figure
            fig_path = self.save_figure(fig, "criteria_distribution")
            prevalence_viz['criteria_distribution'] = fig_path
        
        # --- Visualization 4: Criteria Count Distribution ---
        
        if 'metabolic_criteria_count' in data.columns:
            # Create figure
            fig, ax = plt.subplots(figsize=(self.config['figure_width'], self.config['figure_height']))
            
            # Calculate distribution
            criteria_counts = data['metabolic_criteria_count'].value_counts().sort_index()
            criteria_pct = criteria_counts / len(data) * 100
            
            # Create bar chart
            ax.bar(criteria_counts.index, criteria_pct, color=sns.color_palette(self.config['color_palette'], len(criteria_counts)))
            
            # Add labels
            ax.set_xlabel('Number of Criteria Met')
            ax.set_ylabel('Percentage of Population (%)')
            ax.set_title('Distribution of Metabolic Syndrome Criteria Count', fontsize=14)
            
            # Add value labels
            for i, v in enumerate(criteria_pct):
                ax.text(criteria_counts.index[i], v + 1, f"{v:.1f}%", ha='center', fontsize=10)
            
            # Add grid
            ax.grid(axis='y', alpha=0.3)
            
            # Set x-axis ticks
            ax.set_xticks(range(6))
            
            # Add MetS threshold line
            ax.axvline(x=2.5, color='red', linestyle='--', alpha=0.7)
            ax.text(2.6, max(criteria_pct) * 0.9, 'MetS Threshold (≥3)', color='red', fontsize=10)
            
            # Save figure
            fig_path = self.save_figure(fig, "criteria_count_distribution")
            prevalence_viz['criteria_count_distribution'] = fig_path
        
        # --- Visualization 5: Prevalence Map (Heatmap) ---
        
        # Check if we have at least two demographic variables
        demo_vars = [var for var in ['sex', 'age_group', 'race'] if var in data.columns]
        
        if len(demo_vars) >= 2:
            # Select the first two variables
            var1, var2 = demo_vars[:2]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(self.config['figure_width'], self.config['figure_height']))
            
            # Calculate prevalence by groups
            prevalence_map = data.groupby([var1, var2])['has_metabolic_syndrome'].mean() * 100
            prevalence_map = prevalence_map.unstack()
            
            # Create heatmap
            sns.heatmap(prevalence_map, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
            
            # Add labels
            ax.set_title(f'Metabolic Syndrome Prevalence by {var1.replace("_", " ").title()} and {var2.replace("_", " ").title()}', fontsize=14)
            
            # Save figure
            fig_path = self.save_figure(fig, "prevalence_heatmap")
            prevalence_viz['prevalence_heatmap'] = fig_path
        
        # Store visualizations
        self.visualizations.update(prevalence_viz)
        
        logger.info(f"Created {len(prevalence_viz)} metabolic syndrome prevalence visualizations")
        
        return prevalence_viz
    
    def create_risk_factor_visualizations(self):
        """
        Create visualizations for metabolic syndrome risk factors.
        
        Returns:
            Dictionary with paths to saved visualizations
        """
        logger.info("Creating risk factor visualizations")
        
        # Check if we have the necessary data
        if 'integrated_data' not in self.data and 'clinical_data' not in self.data:
            logger.error("No integrated or clinical data available for risk factor visualizations")
            return {}
        
        # Use integrated data if available, otherwise use clinical data
        data = self.data.get('integrated_data', self.data.get('clinical_data'))
        
        # Check if metabolic syndrome status is available
        if 'has_metabolic_syndrome' not in data.columns:
            logger.error("Metabolic syndrome status not available in data")
            return {}
        
        # Dictionary to store visualization paths
        risk_factor_viz = {}
        
        # --- Visualization 1: Risk Factor Comparison by MetS Status ---
        
        # Define risk factors to compare
        risk_factors = [
            'waist_circumference_cm', 'bmi', 'triglycerides_mg_dL', 'hdl_cholesterol_mg_dL',
            'systolic_bp_mmHg', 'diastolic_bp_mmHg', 'fasting_glucose_mg_dL'
        ]
        
        # Filter to available risk factors
        available_factors = [col for col in risk_factors if col in data.columns]
        
        if not available_factors:
            logger.warning("No risk factors available for comparison visualization")
        else:
            # Create a figure with subplots
            n_factors = len(available_factors)
            n_cols = min(3, n_factors)
            n_rows = (n_factors + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.config['figure_width'], self.config['figure_height'] * n_rows / 2))
            
            # Flatten axes for easier indexing
            if n_rows * n_cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            
            # Create label for MetS status
            data['MetS_Status'] = data['has_<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>