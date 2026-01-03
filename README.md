# Masking Effects on XAI Techniques

## About This Branch

This branch is a compatibility branch designed to work with the [anjana](https://github.com/IFCA-Advanced-Computing/anjana) library for data anonymization. The anjana library has dependency constraints that require older versions of certain packages (particularly numpy). This branch maintains slightly older package versions to ensure compatibility with anjana while still supporting the core functionality of the project.

## Project Overview

This repository contains the code and experiments for a bachelor thesis investigating the effects of data masking and anonymization on Explainable AI (XAI) techniques. The project explores how privacy-preserving data transformations impact the interpretability and performance of machine learning models.

### Key Components

- **Data Anonymization**: Uses anjana for k-anonymity-based data masking
- **Anonymized Preprocessing**: Tools for encoding and transforming anonymized datasets
- **Hierarchy Management**: Generalization hierarchies for quasi-identifiers
- **Experimental Notebooks**: Jupyter notebooks for analyzing different datasets
  - Adult dataset
  - Cervical Cancer dataset
  - USA House dataset

### Project Structure

```
├── src/masking_effects_on_xai_techniques/  # Core Python modules
│   ├── anjana_utils.py                     # Utilities for anjana integration
│   ├── anonymized_preprocessor.py          # Encoding for anonymized data
│   ├── finding_k.py                        # K-anonymity utilities
│   └── hierarchies.py                      # Hierarchy management
├── notebooks/                               # Jupyter notebooks for experiments
├── hierarchies/                             # Generalization hierarchies by dataset
├── data/                                    # Dataset storage
└── hpc/                                     # HPC scripts for large-scale experiments
```

## Dependencies

This branch uses older package versions to maintain compatibility with anjana. See `pyproject.toml` for the complete list of dependencies.
