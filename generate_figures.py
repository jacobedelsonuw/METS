"""
Main script to generate research figures for metabolic syndrome analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import custom modules
from allofus_data_extraction import AllOfUsDataExtractor
from data_preprocessing import DataPreprocessor
from risk_calculator_model import MetabolicSyndromeRiskCalculator
from visualization_generator import MetabolicSyndromeVisualizer
from cluster_analysis import MetabolicSyndromeClusterAnalyzer
from knowledge_representation import MetabolicSyndromeKnowledgeRepresentation
from bayesian_network_analysis import MetabolicSyndromeBayesianNetwork
from knowledge_graph_construction import MetabolicSyndromeKnowledgeGraph

# Create directories for output
os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Initialize data extractor
print("Initializing AllOfUs data extractor...")
extractor = AllOfUsDataExtractor(cohort_size=10000)
extractor.connect_to_database()

# Extract metabolic syndrome components and clinical data
print("Extracting metabolic syndrome data...")
metabolic_components = extractor.extract_metabolic_syndrome_components()
clinical_data = extractor.get_clinical_measurements()
demographics = extractor.get_demographic_data()
medications = extractor.get_medication_data()
lifestyle = extractor.get_lifestyle_data()

# Merge datasets
print("Merging datasets...")
merged_data = pd.merge(demographics, clinical_data, on='participant_id')
merged_data = pd.merge(merged_data, metabolic_components, on='participant_id')
merged_data = pd.merge(merged_data, medications, on='participant_id')
merged_data = pd.merge(merged_data, lifestyle, on='participant_id')

# Save merged data
merged_data.to_csv('data/merged_data.csv', index=False)
print(f"Merged data saved with {len(merged_data)} participants and {len(merged_data.columns)} variables")

# Initialize data preprocessor
print("Preprocessing data...")
preprocessor = DataPreprocessor()
processed_data = preprocessor.preprocess(merged_data)
processed_data.to_csv('data/processed_data.csv', index=False)

# Generate descriptive statistics
print("Generating descriptive statistics...")
stats = preprocessor.generate_descriptive_statistics(processed_data)
stats.to_csv('data/descriptive_statistics.csv')

# Initialize risk calculator
print("Calculating metabolic syndrome risk...")
risk_calculator = MetabolicSyndromeRiskCalculator()
risk_data = risk_calculator.calculate_risk_scores(processed_data)
risk_data.to_csv('data/risk_scores.csv', index=False)

# Train and evaluate risk prediction model
print("Training risk prediction model...")
X_train, X_test, y_train, y_test = risk_calculator.prepare_training_data(processed_data)
model = risk_calculator.train_model(X_train, y_train)
evaluation = risk_calculator.evaluate_model(model, X_test, y_test)
risk_calculator.save_model(model, 'models/risk_prediction_model.pkl')

with open('data/model_evaluation.txt', 'w') as f:
    f.write(f"Model Evaluation Results:\n")
    f.write(f"Accuracy: {evaluation['accuracy']:.4f}\n")
    f.write(f"Precision: {evaluation['precision']:.4f}\n")
    f.write(f"Recall: {evaluation['recall']:.4f}\n")
    f.write(f"F1 Score: {evaluation['f1']:.4f}\n")
    f.write(f"ROC AUC: {evaluation['roc_auc']:.4f}\n")
    f.write(f"\nFeature Importances:\n")
    for feature, importance in evaluation['feature_importances'].items():
        f.write(f"{feature}: {importance:.4f}\n")

# Initialize visualizer
print("Generating visualizations...")
visualizer = MetabolicSyndromeVisualizer(output_dir='figures/visualizations')

# Generate prevalence visualizations
print("Generating prevalence visualizations...")
visualizer.plot_metabolic_syndrome_prevalence(metabolic_components)
visualizer.plot_component_distribution(metabolic_components)
visualizer.plot_component_combinations(metabolic_components)

# Generate demographic visualizations
print("Generating demographic visualizations...")
visualizer.plot_prevalence_by_demographics(metabolic_components, demographics)
visualizer.plot_age_distribution(processed_data)
visualizer.plot_gender_distribution(processed_data)

# Generate clinical measurement visualizations
print("Generating clinical measurement visualizations...")
visualizer.plot_clinical_measurements(clinical_data)
visualizer.plot_measurements_by_metabolic_syndrome(clinical_data, metabolic_components)
visualizer.plot_correlation_heatmap(clinical_data)

# Generate risk visualizations
print("Generating risk visualizations...")
visualizer.plot_risk_distribution(risk_data)
visualizer.plot_risk_by_demographics(risk_data, demographics)
visualizer.plot_roc_curve(evaluation['fpr'], evaluation['tpr'], evaluation['roc_auc'])
visualizer.plot_feature_importance(evaluation['feature_importances'])

# Generate longitudinal visualizations
print("Generating longitudinal visualizations...")
longitudinal_data = extractor.get_longitudinal_data()
visualizer.plot_longitudinal_trends(longitudinal_data)

# Generate summary dashboard
print("Generating summary dashboard...")
visualizer.create_summary_dashboard(
    metabolic_components, 
    clinical_data, 
    risk_data, 
    evaluation
)

# Initialize cluster analyzer
print("Performing cluster analysis...")
cluster_analyzer = MetabolicSyndromeClusterAnalyzer(output_dir='figures/clustering')

# Prepare data for clustering
cluster_features = [
    'age', 'waist_circumference', 'systolic_bp', 'diastolic_bp',
    'hdl_cholesterol', 'triglycerides', 'fasting_glucose', 'bmi'
]
cluster_data = cluster_analyzer.prepare_data(processed_data, features=cluster_features)

# Determine optimal number of clusters
optimal_k = cluster_analyzer.determine_optimal_clusters(cluster_data)
print(f"Optimal number of clusters: {optimal_k}")

# Perform clustering
n_clusters = optimal_k.get('silhouette', 4)  # Use silhouette method, default to 4
cluster_results = cluster_analyzer.perform_clustering(cluster_data, n_clusters=n_clusters)

# Analyze clusters
cluster_analysis = cluster_analyzer.analyze_clusters(
    cluster_results,
    clinical_data,
    metabolic_components
)
cluster_analysis.to_csv('data/cluster_analysis.csv')

# Plot cluster profiles
cluster_analyzer.plot_cluster_profiles(cluster_analysis)

# Plot feature importance
cluster_analyzer.plot_feature_importance_by_cluster()

# Compare clustering methods
cluster_analyzer.compare_clustering_methods(cluster_data, methods=['kmeans', 'hierarchical', 'gmm'])

# Evaluate clustering
cluster_analyzer.evaluate_clustering(cluster_data, cluster_results['cluster'])

# Initialize knowledge representation
print("Creating knowledge representation...")
knowledge_rep = MetabolicSyndromeKnowledgeRepresentation(output_dir='figures/knowledge')

# Create and visualize knowledge graph
knowledge_rep.create_knowledge_graph()
knowledge_rep.visualize_knowledge_graph()

# Visualize component relationships
knowledge_rep.visualize_component_relationships()

# Visualize risk factors
knowledge_rep.visualize_risk_factors()

# Visualize intervention effectiveness
knowledge_rep.visualize_intervention_effectiveness()

# Visualize condition risk
knowledge_rep.visualize_condition_risk()

# Visualize criteria comparison
knowledge_rep.visualize_criteria_comparison()

# Export knowledge base
knowledge_rep.export_knowledge_base('data/knowledge_base.json')

# Export relationships
knowledge_rep.export_relationships('data/relationships.json')

# Create interactive visualization
knowledge_rep.create_interactive_visualization('figures/knowledge/interactive_graph.html')

# Initialize Bayesian network analyzer
print("Creating Bayesian network analysis...")
bn_analyzer = MetabolicSyndromeBayesianNetwork(output_dir='figures/bayesian')

# Define features for discretization
bn_features = [
    'age', 'gender', 'waist_circumference', 'systolic_bp', 'diastolic_bp',
    'hdl_cholesterol', 'triglycerides', 'fasting_glucose', 'metabolic_syndrome'
]

# Define bins for each feature
bins = {
    'age': [0, 30, 50, 70, 100],
    'waist_circumference': 3,
    'systolic_bp': 3,
    'diastolic_bp': 3,
    'hdl_cholesterol': 3,
    'triglycerides': 3,
    'fasting_glucose': 3,
    'metabolic_syndrome': 2  # Binary variable
}

# Define labels for each feature
labels = {
    'age': ['<30', '30-50', '50-70', '70+'],
    'waist_circumference': ['Normal', 'Elevated', 'High'],
    'systolic_bp': ['Normal', 'Elevated', 'High'],
    'diastolic_bp': ['Normal', 'Elevated', 'High'],
    'hdl_cholesterol': ['Low', 'Normal', 'High'],
    'triglycerides': ['Normal', 'Elevated', 'High'],
    'fasting_glucose': ['Normal', 'Elevated', 'High'],
    'metabolic_syndrome': ['No', 'Yes']
}

# Discretize data
discretized_data = bn_analyzer.discretize_data(
    processed_data,
    features=bn_features,
    bins=bins,
    labels=labels
)
discretized_data.to_csv('data/discretized_data.csv', index=False)

# Learn network structure
bn_analyzer.learn_structure(discretized_data)

# Fit parameters
bn_analyzer.fit_parameters(discretized_data)

# Plot network
bn_analyzer.plot_network()

# Query probability of metabolic syndrome
query_result = bn_analyzer.query_probability('metabolic_syndrome')
bn_analyzer.plot_query_result(query_result, 'metabolic_syndrome')

# Query probability of metabolic syndrome given high waist circumference
evidence = {'waist_circumference': 'High'}
query_result = bn_analyzer.query_probability('metabolic_syndrome', evidence)
bn_analyzer.plot_query_result(query_result, 'metabolic_syndrome', evidence)

# Plot Markov blanket of metabolic syndrome
bn_analyzer.plot_markov_blanket('metabolic_syndrome')

# Plot conditional probability table
bn_analyzer.plot_conditional_probability_table('metabolic_syndrome')

# Plot inference results with different evidence
evidence_list = [
    {'waist_circumference': 'High'},
    {'hdl_cholesterol': 'Low'},
    {'fasting_glucose': 'High'},
    {'waist_circumference': 'High', 'hdl_cholesterol': 'Low'}
]
bn_analyzer.plot_inference_results('metabolic_syndrome', evidence_list)

# Plot sensitivity analysis
bn_analyzer.plot_sensitivity_analysis('metabolic_syndrome', 'Yes', bn_features)

# Plot intervention effects
interventions = [
    {'name': 'Weight Loss', 'evidence': {'waist_circumference': 'Normal'}},
    {'name': 'Improved HDL', 'evidence': {'hdl_cholesterol': 'High'}},
    {'name': 'Controlled Glucose', 'evidence': {'fasting_glucose': 'Normal'}},
    {'name': 'Combined Intervention', 'evidence': {'waist_circumference': 'Normal', 'hdl_cholesterol': 'High', 'fasting_glucose': 'Normal'}}
]
bn_analyzer.plot_intervention_effects('metabolic_syndrome', interventions)

# Save model
bn_analyzer.save_model('models/bayesian_network.bif')

# Initialize knowledge graph constructor
print("Creating knowledge graph...")
kg_constructor = MetabolicSyndromeKnowledgeGraph(output_dir='figures/knowledge_graph')

# Create component graph
component_cols = ['elevated_waist', 'elevated_bp', 'reduced_hdl', 'elevated_triglycerides', 'elevated_glucose']
component_data = metabolic_components[component_cols]
kg_constructor.create_component_graph(component_data)

# Create patient similarity graph
kg_features = ['waist_circumference', 'systolic_bp', 'diastolic_bp', 'hdl_cholesterol', 'triglycerides', 'fasting_glucose']
patient_data = pd.merge(
    clinical_data[['participant_id'] + kg_features],
    metabolic_components[['participant_id', 'metabolic_syndrome']],
    on='participant_id'
)
kg_constructor.create_patient_similarity_graph(patient_data, kg_features)

# Create bipartite graph
kg_constructor.create_bipartite_graph(
    patient_data[['participant_id']],
    metabolic_components[['participant_id'] + component_cols]
)

# Create component co-occurrence graph
kg_constructor.create_component_co_occurrence_graph(
    metabolic_components[['participant_id'] + component_cols]
)

# Create risk factor graph
risk_factors = ['age_over_50', 'male_gender', 'obesity', 'smoking', 'physical_inactivity', 'family_history']
# Create risk factor variables
processed_data['age_over_50'] = (processed_data['age'] > 50).astype(int)
processed_data['male_gender'] = (processed_data['gender'] == 'Male').astype(int)
processed_data['obesity'] = (processed_data['bmi'] >= 30).astype(int)

kg_constructor.create_risk_factor_graph(processed_data, risk_factors)

# Create community graph
kg_constructor.create_community_graph(patient_data, kg_features)

# Create interactive visualization
kg_constructor.create_interactive_visualization('figures/knowledge_graph/interactive_graph.html')

# Export graph
kg_constructor.export_graph('data/knowledge_graph.json')

print("All research figures generated successfully!")
print(f"Total figures generated: {len(os.listdir('figures/visualizations')) + len(os.listdir('figures/clustering')) + len(os.listdir('figures/knowledge')) + len(os.listdir('figures/bayesian')) + len(os.listdir('figures/knowledge_graph'))}")

# Generate timestamp
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open('data/generation_log.txt', 'w') as f:
    f.write(f"Research figures generated on: {timestamp}\n")
    f.write(f"Total participants: {len(processed_data)}\n")
    f.write(f"Metabolic syndrome prevalence: {processed_data['metabolic_syndrome'].mean():.2%}\n")
    f.write(f"Optimal number of clusters: {n_clusters}\n")
    f.write(f"Risk model accuracy: {evaluation['accuracy']:.4f}\n")
    f.write(f"Total figures generated: {len(os.listdir('figures/visualizations')) + len(os.listdir('figures/clustering')) + len(os.listdir('figures/knowledge')) + len(os.listdir('figures/bayesian')) + len(os.listdir('figures/knowledge_graph'))}\n")

print("Generation log saved to data/generation_log.txt")
