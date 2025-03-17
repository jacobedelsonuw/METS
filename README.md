# Metabolic Syndrome Research Project

This repository contains a comprehensive set of Python modules for analyzing metabolic syndrome using the All of Us Research Program database.

## Project Structure

- `allofus_data_extraction.py`: Extracts and processes data from the All of Us Research Program
- `data_preprocessing.py`: Handles data cleaning, transformation, and feature engineering
- `risk_calculator_model.py`: Implements risk prediction models for metabolic syndrome
- `visualization_generator.py`: Creates visualizations for metabolic syndrome analysis
- `cluster_analysis.py`: Performs cluster analysis to identify patient subgroups
- `knowledge_representation.py`: Represents knowledge about metabolic syndrome
- `bayesian_network_analysis.py`: Analyzes probabilistic relationships between components
- `knowledge_graph_construction.py`: Constructs knowledge graphs for metabolic syndrome
- `generate_figures.py`: Orchestrates the entire analysis pipeline

## Usage

To run the complete analysis pipeline:

```bash
python generate_figures.py
```

This will:
1. Extract data from the All of Us database
2. Preprocess the data
3. Calculate risk scores
4. Generate visualizations
5. Perform cluster analysis
6. Create knowledge representations
7. Build Bayesian networks
8. Construct knowledge graphs
9. Save all figures to the `figures/` directory

## Module Descriptions

### AllOfUs Data Extraction

The `allofus_data_extraction.py` module provides functions to extract and process data from the All of Us Research Program database. It includes:

- Extraction of metabolic syndrome components
- Retrieval of clinical measurements
- Collection of demographic data
- Access to medication information
- Gathering of lifestyle data
- Longitudinal data extraction

### Data Preprocessing

The `data_preprocessing.py` module handles data cleaning, transformation, and feature engineering. It includes:

- Missing value imputation
- Categorical feature encoding
- Numerical feature standardization
- Derived feature creation
- Descriptive statistics generation

### Risk Calculator Model

The `risk_calculator_model.py` module implements risk prediction models for metabolic syndrome. It includes:

- Risk score calculation based on clinical criteria
- Machine learning model training for risk prediction
- Model evaluation metrics
- Feature importance analysis
- Risk visualization functions

### Visualization Generator

The `visualization_generator.py` module creates visualizations for metabolic syndrome analysis. It includes:

- Prevalence visualizations
- Component distribution plots
- Demographic analysis charts
- Clinical measurement visualizations
- Risk distribution plots
- Longitudinal trend analysis
- Summary dashboards

### Cluster Analysis

The `cluster_analysis.py` module performs cluster analysis to identify patient subgroups. It includes:

- Data preparation for clustering
- Optimal cluster number determination
- Multiple clustering algorithms (K-means, hierarchical, GMM)
- Cluster profiling and characterization
- Cluster evaluation metrics

### Knowledge Representation

The `knowledge_representation.py` module represents knowledge about metabolic syndrome. It includes:

- Formal knowledge representation framework
- Component relationship modeling
- Risk factor representation
- Intervention effectiveness modeling
- Criteria comparison functions

### Bayesian Network Analysis

The `bayesian_network_analysis.py` module analyzes probabilistic relationships between components. It includes:

- Data discretization
- Network structure learning
- Parameter estimation
- Probabilistic inference
- Sensitivity analysis
- Intervention modeling

### Knowledge Graph Construction

The `knowledge_graph_construction.py` module constructs knowledge graphs for metabolic syndrome. It includes:

- Component graph creation
- Patient similarity graph construction
- Bipartite graph modeling
- Co-occurrence graph analysis
- Risk factor graph development
- Community detection

## Requirements

- Python 3.10+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- networkx
- pgmpy
- pyvis

# Metabolic Syndrome Risk Calculator

This project is a React-based application that calculates the risk of metabolic syndrome based on clinical measurements. It provides a user-friendly interface to input patient data, assess risk, and visualize results.

## Features

- Input patient demographics and core measurements
- Calculate risk of metabolic syndrome
- Visualize risk assessment results
- Load predefined patient profiles for quick assessment
- Reset form to default values

## Getting Started

### Prerequisites

- Node.js
- npm or yarn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/metabolic_calculator.git
   cd metabolic_calculator/my-metabolic-calculator
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

### Running the Application

1. Start the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

2. Open your browser and navigate to `http://localhost:3000`.

### Usage

1. Input patient data using the sliders and switches.
2. Click the "Calculate Risk" button to assess the risk of metabolic syndrome.
3. View the risk assessment results in the "Risk Assessment" tab.
4. Use the "Visualization" tab to see a graphical representation of the results.
5. Use the "Intervention" tab to input lifestyle intervention data and recalculate the risk.
6. Load predefined patient profiles using the buttons in the footer.
7. Reset the form to default values using the "Reset" button.
