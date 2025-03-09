#  🌊 AQUA-QUANT 🪸
**A**rtificial **I**ntelligence and **QU**antum **A**pproach for **QU**antitative **A**ssessment of **N**atural **T**rends: Accelerating Marine Ecosystem Recovery with Quantum Computing





# Quantum Evolution Project

This project demonstrates the use of quantum-enhanced machine learning tools with the [Quantum Evolution Kernel](https://github.com/pasqal-io/quantum-evolution-kernel) library along with popular Python data science libraries.

## Overview

The project leverages several libraries:
- **numpy**: For numerical operations.
- **pandas**: For data manipulation.
- **scikit-learn**: For machine learning tools.
- **matplotlib** & **seaborn**: For data visualization.
- **pytorch**: For deep learning frameworks.
- **qadence**: (Verify the package name if installation issues arise.)
- **pulser**: For quantum control tasks.
- **quantum-evolution-kernel**: For quantum-enhanced graph machine learning.

> **Note:** The `quantum-evolution-kernel` package requires Python 3.10 or later.


# Quantum-Enhanced Coral Bleaching Prediction System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Pulser](https://img.shields.io/badge/Pulser-latest-orange)](https://github.com/pasqal-io/Pulser)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

> **Blaise Pascal Quantum Challenge Project**: A quantum-enhanced machine learning system for early prediction of coral bleaching events using Pasqal's neutral atom quantum processors.

## 🌊 Project Overview

Coral reefs are experiencing unprecedented bleaching events due to climate change. Our system leverages quantum computing and AI to predict these events 6-8 weeks in advance, providing critical additional time for implementing protection measures.

This repository contains the code and documentation for a hybrid quantum-classical model that uses Pasqal's neutral atom quantum processors to enhance the prediction capabilities beyond what is possible with classical machine learning alone.

## 🔑 Key Features

- **Quantum Embedding Kernel (QEK)** implementation for oceanographic data
- **Hybrid quantum-classical ML pipeline** combining quantum feature mapping with classical SVM
- **Advanced data preprocessing** for multiple oceanographic parameters
- **Benchmarking framework** comparing quantum vs. classical approaches
- **Visualization tools** for prediction results and model interpretation

## 🧪 Minimal Working Example

The repository includes a complete Minimal Working Example (MWE) demonstrating:

1. Processing of historical SST and bleaching data for the Great Barrier Reef
2. Implementation of quantum feature mapping using Pulser
3. Quantum kernel computation and integration with classical SVM
4. Evaluation and comparison with classical ML approaches

## Environment Setup

A Conda environment YAML file is provided to simplify the setup. Follow these steps to create your environment:

1. **Ensure Conda is installed:**  
   You can install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

2. **Create the environment:**  
   Save the provided `environment.yml` file in your project root and run:
   ```bash
   conda env create -f environment.yml


## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/pasqal-io/Pasqal_Hackathon_Feb25_Team_15.git
cd Pasqal_Hackathon_Feb25_Team_15

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install notebook

# Run the notebook containing the MVE 
cd test
jupyter notebook Hybrid_Coral_Reef_Bleaching.ipynb
```

## 📊 Data Sources

The project uses the following data sources:
* [NOAA Reef Check & Coral Bleaching Data](https://www.kaggle.com/datasets/oasisdata/noaa-reef-check-coral-bleaching-data).
