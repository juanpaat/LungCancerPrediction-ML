# Lung Cancer Prediction using ML

## Project Overview
Lung cancer is one of the leading causes of cancer-related deaths worldwide. This project aims to develop predictive models for assessing lung cancer risk based on health, environmental, and lifestyle factors. The goal is to assist doctors, researchers, and data scientists in identifying patterns that improve early detection and intervention strategies.

## Dataset Description
The dataset used in this project comes from Kaggle: [Lung Cancer Risk and Prediction Dataset](https://www.kaggle.com/datasets/ankushpanday1/lung-cancer-risk-and-prediction-dataset). It includes information about various demographic, medical, and lifestyle factors that may contribute to lung cancer risk.

### Features
The dataset consists of 25 features:

1. **Age** - Patient's age (years)
2. **Gender** - Male/Female
3. **Smoking** - Smoking habit (Yes/No)
4. **Yellow Fingers** - Yellowing of fingers (Yes/No)
5. **Anxiety** - Anxiety levels (Yes/No)
6. **Peer Pressure** - Influence of peers on smoking habits (Yes/No)
7. **Chronic Disease** - Presence of chronic diseases (Yes/No)
8. **Fatigue** - Experience of frequent fatigue (Yes/No)
9. **Allergy** - Presence of allergies (Yes/No)
10. **Wheezing** - Wheezing symptoms (Yes/No)
11. **Alcohol Consumption** - Regular alcohol intake (Yes/No)
12. **Coughing** - Persistent coughing (Yes/No)
13. **Shortness of Breath** - Breathing difficulties (Yes/No)
14. **Swallowing Difficulty** - Difficulty swallowing (Yes/No)
15. **Chest Pain** - Occurrence of chest pain (Yes/No)
16. **Balanced Diet** - Adherence to a healthy diet (Yes/No)
17. **Exercise Frequency** - Frequency of physical exercise (Yes/No)
18. **Genetic Risk** - Family history of lung cancer (Yes/No)
19. **Exposure to Pollution** - Living in polluted areas (Yes/No)
20. **Occupational Hazards** - Exposure to workplace carcinogens (Yes/No)
21. **Sleep Quality** - Sleep quality score (1-10)
22. **BMI** - Body Mass Index
23. **Blood Pressure** - Blood pressure levels
24. **Diabetes** - Diabetes diagnosis (Yes/No)
25. **Lung Cancer** - Target variable indicating lung cancer diagnosis (Yes/No)

## Repository Structure
```
├── lung_cancer_prediction.ipynb  # Jupyter Notebook for data processing & ML modeling
├── lung_cancer_prediction.csv    # Dataset used for training and testing models
├── README.md                     # Project documentation
├── requirements.txt              # Requirements to run the script
```

## Installation & Dependencies
To set up the environment and install the required dependencies, follow these steps:

```bash
# Clone the repository
git clone https://github.com/juanpaat/LungCanderPrediction-ML.git

cd LungCanderPrediction-ML

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

## Data Preprocessing
Key preprocessing steps performed in the notebook:
- **Handling missing values**: Checked and imputed missing values if necessary.
- **Encoding categorical variables**: Converted categorical data to numerical format using label encoding.
- **Balancing the dataset**: (SMOTE) Synthetic Minority Over-sampling Technique 

## Machine Learning Models
Several machine learning algorithms were tested:
- Decision Trees
- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naïve Bayes
- Support Vector Machine (SVM)
- XGBoost

### Evaluation Metrics
The models were evaluated using:
- **Accuracy**: Percentage of correct predictions.
- **Precision**: True positives / (True positives + False positives).
- **Recall**: True positives / (True positives + False negatives).

## Results & Findings
- **Random Forest and XGBoost** yielded the highest accuracy.
- **Smoking, genetic risk, and exposure to pollution** were among the most significant features contributing to lung cancer risk.
- **Patients with chronic diseases and frequent fatigue** showed a higher correlation with lung cancer.

## How to Use
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook lung_cancer_prediction.ipynb
   ```
2. Run all cells to preprocess data, train models, and evaluate performance.
3. Modify hyperparameters or algorithms for further experimentation.

## Future Improvements
- Hyperparameter tuning using GridSearchCV.
- Incorporation of deep learning models like neural networks.
- Collecting and integrating a larger, more diverse dataset.
- Feature engineering to improve model performance.

## Contributions & License
Contributions are welcome! If you'd like to contribute, feel free to fork the repository, make modifications, and submit a pull request.

**License:** This project is open-source.

---
**Author:** JuanBioData  
**Contact:** juanpabloalzatetamayo@gmail.com  
**GitHub:** [juanpaat](https://github.com/juanpaat)

