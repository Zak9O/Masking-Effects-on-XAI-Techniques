# Masking Effects on XAI Techniques

## About This Branch

This branch preprocess and anonymized datasets. It is a compatibility branch designed to work with the [anjana](https://github.com/IFCA-Advanced-Computing/anjana) library for data anonymization. The anjana library has dependency constraints that require older versions of certain packages (particularly numpy). This branch maintains slightly older package versions to ensure compatibility with anjana while still supporting the core functionality of the project.

A fork of the anjana library has been created by the authors of this project, since the anjana library does not support working with continuous features. 

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

## Specs of the explanation jobs run on HPC cluster

- 1 Core with clockspeed in 2.8-3.4GHz range
- 1 GB of memory
