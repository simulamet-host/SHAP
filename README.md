# SHAP: Explainability of Machine Learning Models under Missing Data

This repository contains the official codes for the paper **"Explainability of Machine Learning Models under Missing Data"**.

## Goal

The purpose of this research is to evaluate the impact of different data imputation methods on the explainability of machine learning models. Specifically, we introduce missing data into various datasets, apply several imputation techniques (e.g., Mean Imputation, MICE, missForest, GAIN), and then train XGBoost models.

We use **SHAP (SHapley Additive exPlanations)** to explain the predictions of these models. By comparing the SHAP values from models trained on imputed data to the SHAP values from a model trained on the original, complete data, we can quantify how well each imputation method preserves model explainability.

## Directory Structure

For the code to run correctly, your project should follow this structure. The `funcs` directory contains the implementations for the different imputation methods and utility functions used in the notebooks.

```
.
├── funcs/
│   ├── utils.py           # Utility functions (e.g., generate_missing_data)
│   ├── explain.py         # Main helper functions for running experiments
│   ├── explainNumpy.py    # Helper functions for numpy data (used in MNIST)
│   ├── DIMV.py            # DIMV imputation implementation
│   ├── miss_forest.py     # missForest imputation implementation
│   └── GAIN/
│       └── gain.py        # GAIN imputation implementation
├── results/               # Directory where output plots/tables are saved
│
├── XGB_clf_glass_rate 02.ipynb
├── XGBRegressor_california_rate02.ipynb
├── XGBRegressor_diabetes_rate02.ipynb
├── XGB mnist with GAIN 02.ipynb
└── README.md
```

**Note:** You must create the `results/` directory yourself before running the notebooks.

## Setup & Installation

It is highly recommended to use a virtual environment (like `conda` or `venv`) to manage dependencies.

1.  **Create and activate a new virtual environment.**

2.  **Install the required libraries.** The main dependencies are:

      * `jupyter`
      * `scikit-learn`
      * `numpy`
      * `pandas`
      * `seaborn`
      * `matplotlib`
      * `shap`
      * `xgboost`
      * `tensorflow` (for GAIN imputation)
      * `torch` & `torchvision` (for the MNIST experiment)

    You can install them via pip:

    ```bash
    pip install jupyter sklearn numpy pandas seaborn matplotlib shap xgboost tensorflow torch torchvision
    ```

3.  **Ensure the `funcs` directory is in the same root folder as the notebooks.** The notebooks import imputation methods and helpers directly from this folder.

## Running the Experiments

Each Jupyter Notebook (`.ipynb`) represents a self-contained experiment on a specific dataset. To keep the repo clean, we upload the notesbook for only 20\% missing rate. However, higher missing rates can be easily achieved by changing the parameter missing_rate in each notebook.


### Experiment Notebooks:

1.  **`XGB_clf_glass_rate 02.ipynb`**

      * **Task:** Classification
      * **Model:** `xgboost.XGBClassifier`
      * **Dataset:** Glass Identification (downloaded automatically from the UCI ML repository)
      * **Imputers:** Mean, MICE, DIMV, missForest, SOFT-IMPUTE, GAIN.

2.  **`XGBRegressor_diabetes_rate02.ipynb`**

      * **Task:** Regression
      * **Model:** `xgboost.XGBRegressor`
      * **Dataset:** Diabetes (loaded from `sklearn.datasets`)
      * **Imputers:** Mean, MICE, DIMV, missForest, SOFT-IMPUTE, GAIN.

3.  **`XGBRegressor_california_rate02.ipynb`**

      * **Task:** Regression
      * **Model:** `xgboost.XGBRegressor`
      * **Dataset:** California Housing (loaded from `shap.datasets`)
      * **Imputers:** Mean, MICE, DIMV, missForest, SOFT-IMPUTE, GAIN.

4.  **`XGB mnist with GAIN 02.ipynb`**

      * **Task:** Classification
      * **Model:** `xgboost.XGBClassifier`
      * **Dataset:** MNIST (downloaded automatically via `torchvision.datasets`)
      * **Imputers:** Mean, MICE, DIMV, missForest, SOFT-IMPUTE, GAIN.

### Workflow

Each notebook follows a similar workflow:

1.  **Load Data:** The dataset is loaded (either from a library or downloaded).
2.  **Preprocessing:** Data is split into training/test sets and standardized.
3.  **Generate Missingness:** A copy of the data is created with a 20% missing rate (`X_train_star`, `X_test_star`).
4.  **Imputation:** The data with missing values is imputed using various methods.
5.  **Model Training:** An XGBoost model is trained on the original (complete) data and on each of the imputed datasets.
6.  **SHAP Analysis:** SHAP values are calculated for the test set predictions of each model.
7.  **Evaluation:** The "MSE Shap" (Mean Squared Error between SHAP values from the imputed model and the original model) is calculated to measure the impact on explainability. Other metrics like prediction MSE are also computed.
8.  **Results:** Tables and plots comparing the performance and explainability metrics are generated and saved to the `results/` folder.
