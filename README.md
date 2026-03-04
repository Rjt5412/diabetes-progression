# Diabetes Progression Risk Prediction

A machine learning project to predict the risk of diabetes progression over a 1-year period using clinical patient data. This repository includes the full research pipeline (notebooks), data, model artifacts, and a production-ready Streamlit dashboard.

## Project Structure

- `data/`: Raw and processed dataset files.
- `notebooks/`: 
  - `01_eda.ipynb`: Exploratory Data Analysis and Treatment of missing values.
  - `02_preprocessing_and_modeling.ipynb`: Pipeline construction and SVM hyperparameter tuning.
  - `03_evaluation_and_fairness.ipynb`: Performance validation and demographic fairness audit.
  - `04_explainability_and_drift.ipynb`: SHAP interpretation and monitoring setup.
- `artifacts/`: Serialized models, scalers, and monitoring metadata.
- `src/`: 
  - `app.py`: Streamlit-based web interface for real-time risk prediction.
- `modeling_report.md`: Comprehensive documentation of the modeling approach and results.
- `Dockerfile`: Containerization setup for deployment.

## Getting Started

1. **Environment Setup:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Training the Model

The model training process is documented and executed through a series of Jupyter Notebooks. To reproduce the training:

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Execute the notebooks in the following order:
   - **Step 1: EDA**: `notebooks/01_eda.ipynb` (Handles data cleaning and invalid zeros).
   - **Step 2: Modeling**: `notebooks/02_preprocessing_and_modeling.ipynb` (Trains the SVM and computes the optimal decision threshold).
   - **Step 3: Evaluation**: `notebooks/03_evaluation_and_fairness.ipynb` (Performs calibration and fairness audits).
   - **Step 4: Explainability**: `notebooks/04_explainability_and_drift.ipynb` (Generates global/local SHAP values and drift baselines).

Models are automatically saved to the `artifacts/` directory upon successful training.

## Running the Application

Once the environment is set up and requirements are installed, launch the interactive dashboard:

### Local Development
```bash
streamlit run src/app.py
```

### With Docker
1. Build the image:
```bash
docker build -t diabetes-risk-app .
```
2. Run the container:
```bash
docker run -p 8501:8501 diabetes-risk-app
```

## Model Details
- **Architecture:** SVM with RBF Kernel (Probability enabled)
- **Primary Metrics:** F0.5 Score (prioritizing Precision for clinical risk management).
- **Explainability:** SHAP KernelExplainer used for interactive patient-level risk narratives.
- **Fairness:** Audited across Age and BMI segments with demographic parity checks.

## Dashboard Demo

![App Demo](./demo.gif)
