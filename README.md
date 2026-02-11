# Machine Learning Final Project: Lung Cancer Diagnosis & Life Expectancy

"This project represents the culmination of our Machine Learning specialization, featuring an advanced Dynamic Survival Analysis system for lung cancer patients. Moving beyond static predictions, we developed an end-to-end architecture that integrates clinical oncology data with socioeconomic determinants. Our solution utilizes a multi-step Gradient Boosting framework to estimate survival probabilities across five-year milestones, providing a high-fidelity tool for clinical decision support."

> *"Hard work always beats talent when talent doesn't work hard"* - Tim Notke

## 👥 Credits

> - [Elius Trujillo](https://github.com/elius123ef)


## 🎯 Project Goal

The goal of this project is to develop a comprehensive machine learning solution to:
- Process and clean complex oncology datasets (SEER).
- Perform extensive exploratory data analysis (EDA) to find correlations between income and survival.
- Train and optimize a set of gradient reinforcement models (XGBoost) to predict a patient's 5-year survival probability.
- Visualize the risk of refusing medical recommendations.
- Provide a tool for clinical and social analysis using machine learning techniques.

## 🚀 Project Overview

### Problem Statement
Lung cancer prognosis is traditionally focused on clinical staging, often overlooking the critical role of socioeconomic barriers. While clinical data defines the disease, Household Income often dictates the patient's path to life-saving interventions like surgery.

The challenge lies in the lack of tools that translate complex oncology data into dynamic trajectories. This project addresses the need for a system that doesn't just predict a single number of months, but quantifies how socioeconomic status and treatment decisions (or the rejection of them) dynamically alter a patient's probability of survival over a 5-year horizon

### Dataset
We utilized the **SEER (Surveillance, Epidemiology, and End Results)** dataset, focusing on Lung cancer cases (2012-2022).
- **Instances:** 60,000+ records.
- **Predictors:** Initially, the dataset consisted of a total of 21 predictor variables, which, after combining them and cleaning up values ​​that do not contribute significant importance, have been reduced to a total of nine: age_group, tumor_category, income_level, Total number of in situ/malignant tumors for patient, tratamiento, Stage_Final, Sex, grade_clinical, histology_type_named.

### Methodology
We transitioned from a single regression approach to a Multi-Milestone Classification Pipeline. This allowed us to capture the non-linear risk of mortality at different stages of the disease:

Model Architecture: We implemented 5 independent XGBoost models, each optimized to predict the probability of being alive at specific intervals (12, 24, 36, 48, and 60 months).

Dynamic Data Filtering: For each milestone, we applied a temporal filter to handle censored data, ensuring that only patients with confirmed outcomes or sufficient follow-up time were included.

Feature Engineering: We consolidated complex SEER cancer stages into simplified clinical categories and mapped treatment combinations to analyze the impact of surgical intervention and household income as primary predictors.

Monotonic Logic: A post-processing layer was added to ensure the survival curve consistently decreases over time, maintaining clinical coherence.

### Results
The final **XGBoost** model achieved:
The multi-milestone architecture achieved a robust performance, with accuracy improving as the timeline progressed. However, the first 12 months represent the highest predictive challenge:
Year 1 Performance (12-month Milestone):
Accuracy: 0.79.
Class 1 (Alive) F1-Score: 0.83.
Class 0 (Deceased) F1-Score: 0.75.
Key Findings: While predicting early mortality (Year 1) is complex, the model maintains a high precision (0.82) for survivors.

The model's predictive power increases significantly across milestones. While the first year is highly volatile, the 5-year prediction achieves superior stability, especially in identifying long-term survivors:
As seen in the classification report, the model reaches an accuracy of 0.88. Although the F1-score for the "Alive" class (1) is lower (0.68) due to the natural reduction of the survivor sample (support: 10,000), the Precision (0.75) remains solid, ensuring that predictions of long-term survival are reliable.

## 📝 Project Phases

### Step 1: Problem Definition
The project addresses the need for a personalized prognosis. We transform raw clinical records into a supervised classification problem whose objective is the "annual survival probability".

### Step 2: Acquiring and Loading
Data was sourced from the SEER database, processed into a structured format, and loaded using Pandas for initial cleaning.

### Step 3: Data Acquisition and Access Control 
English: Unlike public datasets, access to the SEER (Surveillance, Epidemiology, and End Results) database required a formal research authorization.

API & SEER*Stat: Data was acquired following a credentialed request to the National Cancer Institute (NCI). We utilized the SEER*Stat API and software to securely extract oncological records.

Data Integrity: We implemented a custom workflow to filter relevant lung cancer events, ensuring that only cases with verified survival metrics and clinical consistency were stored for the EDA phase.

### Step 4: Descriptive Analysis
The descriptive analysis revealed that survival is primarily governed by two dominant factors: Tumor Stage and Treatment Protocol.

Clinical Dominance: We identified that the Tumor Stage at the time of diagnosis is the most powerful predictor of survival, showing a drastic drop in probability between localized and distant stages.

Treatment Synergy: Our analysis confirmed that the combination of treatments (Surgery, Chemotherapy, and Radiation) significantly alters the survival trajectory. Patients undergoing surgical interventions consistently show superior survival rates across all milestones compared to non-surgical protocols.

### Step 5: Full EDA
During the Full EDA phase, we performed deep multivariate analysis to uncover hidden patterns and ensure data quality for modeling:

Correlation Mapping: We transformed the 'survival months' variable into discrete categorical ranges (e.g., 0, 1-5, and 5-year increments) to visualize clear correlations between socioeconomic status and clinical outcomes.

Feature Relationship: We identified that the impact of Income Level is most visible in early diagnosis; higher income correlates with localized stages, which in turn leads to more surgical interventions.

Data Preparation: We handled missing values and encoded categorical variables (Stage, Treatment, Histology) to prepare a clean dataset for the XGBoost architecture, ensuring an 80/20 train-test split for robust validation.

### Step 6: Build the Model and Optimize It
The core of our solution is a Multi-Stage Gradient Boosting Architecture. Instead of a single predictor, we developed a battery of specialized models to capture the changing risk profile of patients over time:

Algorithm Selection: We chose XGBoost due to its superior handling of categorical variables (like Treatment and Stage) and its ability to model non-linear relationships without intensive scaling.

Multi-Milestone Strategy: We trained 5 independent classification models for the 12, 24, 36, 48, and 60-month marks. This approach allows the system to provide a dynamic probability curve rather than a static point estimate.

Hyperparameter Tuning: Each model was optimized using GridSearchCV and RandomizedSearchCV, focusing on max_depth to prevent overfitting and learning_rate (eta) to ensure stable convergence.

Validation: We prioritized the F1-Score and Precision to ensure that the model is reliable when predicting survival, particularly in the later milestones where accuracy reached 0.88.

### Step 7: Deploy the Model
*The model is prepared for deployment as a web service where users can input patient profiles to receive a survival estimation.*

## 📁 Project Structure

sp-ml-20-final-project-g1

├── 📁 data/

│ ├── 📁 interim/ # Intermediate transformed data 

│ ├── 📁 processed/ # Final data used for modeling

│ └── 📁 raw/ # Raw data without processing 

├── 📁 database/ # SQL scripts and database configs 

├── 📁 docs/

│ └── Presentacion.pdf # Project presentation 

├── 📁 models/

│ └── pipelines.pkl # Trained XGBoost model

├── 📁 notebooks/

│ ├── EDA_FINAL.ipynb # Exploratory Data Analysis and Feature Engineering

│ └──MODELO_XGB.ipynb # Modeling

├── 📁 src/

│ └── predict_survival.py # Prediction engine script 

├── 📁 webapp/ # Deployment application (Streamlit/Flask)

├── 📁 .streamlit/

│   └── config.toml           # Streamlit theme and UI configuration

├── 📁 assets/                # Visual resources and multimedia

│   ├── betania.jpg           # Team member profile image

│   ├── carlos.jpg            # Team member profile image

│   ├── elius.jpg             # Team member profile image

│   └── referencia_anatomica.PNG # Anatomical reference image for clinical context

├── 📁 data/                  # Project datasets

│   ├── EDA_FINAL.csv         # Processed dataset for exploratory analysis

│   └── referencia_pacientes.csv # Patient reference metadata

├── 📁 model/                 # Pre-trained Machine Learning pipelines

│   ├── pipeline_12m.pkl      # 12-month survival prediction model

│   ├── pipeline_24m.pkl      # 24-month survival prediction model

│   ├── pipeline_36m.pkl      # 36-month survival prediction model

│   ├── pipeline_48m.pkl      # 48-month survival prediction model

│   └── pipeline_60m.pkl      # 60-month survival prediction model

├── 📁 pages/                 # Application modules and navigation

│   ├── 1_📚_Metodologia.py    # Methodology and scientific approach

│   ├── 2_📊_EDA.py            # Exploratory Data Analysis dashboard

│   ├── 3_🧠_Prediccion.py     # Core inference and prediction interface

│   ├── 4_⚕️_Escenarios_Tratamiento.py # Treatment scenario simulator

│   ├── 5_📈_Rendimiento_Modelos.py    # Model performance metrics (DataFrame format)

│   ├── 6_📌_Conclusiones.py    # Key findings and project summary

│   └── 7_👤_Equipo.py         # Team credits and contact info

├── Home.py                  # Main entry point for the Streamlit application

├── README.md                # Project documentation

└── utils.py                 # Helper functions and global design settings


## 🛠️ Technologies Used
- **Python:** Primary language.
- **Pandas & Numpy:** Data manipulation.
- **XGBoost & Scikit-Learn:** Machine Learning modeling.
- **Matplotlib & Seaborn:** Data visualization.
- **Joblib:** Model serialization.

## 📊 Results Summary
The multi-milestone architecture demonstrated high reliability in predicting long-term survival, showing that while early-stage prognosis is volatile, the model stabilizes significantly at the 5-year mark.

Model Performance:
Year 1 (Short-term): Achieved 79% Accuracy. The challenge in this milestone reflects the biological aggression of lung cancer in its initial phase.
Year 5 (Long-term): Achieved an outstanding 88% Accuracy, providing high confidence for long-term clinical planning.
Key Clinical Insights:
The surgical factor: Surgical intervention was identified as the most influential non-biological predictor, capable of increasing the probability of 5-year survival by a high percentage in the localized stage.

## 🌐 Live Demo
*[Link to be added upon deployment]*
