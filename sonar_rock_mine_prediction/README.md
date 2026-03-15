# Sonar Rock vs Mine Prediction

## Overview

This project builds a Machine Learning model to classify objects detected by SONAR signals as either **Rock** or **Mine**. The model analyzes sonar frequency energy patterns and learns to distinguish between the two classes.

This is a **binary classification problem** where the model predicts whether an object under water is a rock or a mine.

---

## Dataset

The dataset used in this project contains **60 numerical features** representing sonar signal energy at different frequency bands.

* Total Samples: 208
* Features: 60
* Target Classes:

  * `R` → Rock
  * `M` → Mine

Each row represents the sonar signal returned from an object.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn

---

## Machine Learning Model

The model used in this project is **Logistic Regression** for binary classification.

### Workflow

1. Load dataset
2. Data preprocessing
3. Train-test split
4. Model training
5. Model evaluation
6. Prediction on new data

---

## Model Training

Dataset split:

* Training Data: 80%
* Testing Data: 20%

Model trained using:

```python
from sklearn.linear_model import LogisticRegression
```

---

## Model Performance

Example accuracy achieved:

```
Training Accuracy: ~80%
Testing Accuracy: ~80%
```

Accuracy may vary depending on random split.

---

## Example Prediction

Input sonar signal values:

```
(0.0200,0.0371,0.0428,0.0207,...)
```

Model prediction:

```
This is Rock
```

or

```
This is Mine
```

---

## Project Structure

```
sonar_rock_mine_prediction
│
├── sonar data.csv
├── sonar_rock_mine_prediction.ipynb
├── sonar_rock_mine_prediction.py
└── README.md
```

---

## How to Run

1. Clone the repository

```
git clone https://github.com/aashutoshsinha99/ai_ml_projects.git
```

2. Install dependencies

```
pip install pandas numpy scikit-learn
```

3. Run the notebook or Python script.

---

## Author

Aashutosh Sinha
