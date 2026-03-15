#!/usr/bin/env python
# coding: utf-8

import pandas as pd

# Read the dataframe, header isnt there so we init with None
df = pd.read_csv("sonar data.csv", header=None) 
# Print first 5 rows
df.head()


# Check if any cell is null?
df.isnull().sum()


# separate input X and target Y
X = df.drop(columns=60) # 60th column has the output basically, so we removed it from X
Y = df[60]


# So now we have the output Y.
# Lets divide our input X (samples) into training and test data
# test data should be 20 % (test_size=0.2)
# Class separation info (Rock class and Mine class) is available in Y.
# So use that (stratify=Y) and keep the same ratio for training data as well.

# import the lib 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# X_train has 166 inputs ,  Y train has 166 outputs
# X_test has 42 inputs , Y test has 42 outputs
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# the data looks in same range, so we wont do any preprocessing (standarization)


# Now, we need to determine if the sample row is Rock class or Mine class,
# we can use Logistic regression AI Model which is bets suited for this scenario
# (it is a linear model)

from sklearn.linear_model import LogisticRegression
logistic_regression_model = LogisticRegression()

#let ai learn what to process(X_train) and how it's output will look(Y_train)
logistic_regression_model.fit(X_train, Y_train)


# lets check the accuracy of training data first (optional)

# from sklearn.metrics import accuracy_score
# predicted_y_for_training_data = logistic_regression_model.predict(X_train)
# print(accuracy_score(predicted_y_for_training_data, Y_train))

# instead of predicting and checking accuracy, we can use score().
# score() method internally calls predict and accuracy_score()
print(logistic_regression_model.score(X_train, Y_train))


# lets check the accuracy of testing data
print(logistic_regression_model.score(X_test, Y_test))


# COOL
# Lets try to predict 1 manually just out of curiosity
# Open the sonar "sonar data.csv" file in notepad and choose any 1 row data.
# Copy the entire row except the last row (which contains the answer)
# Lets put that copied data into our machine learnign model as input
# and lets see what it predicts.

input_data = (0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032)

# Scikit-learn models expect a 2-D input (1,60), but our input_data passed is a 1-D tuple.(60,)
# change 1-D to 2-D using numpy
import numpy as np
input_data_np = np.asarray(input_data)
# 1  → one sample
# -1 → automatically infer number of features
input_data_reshaped = input_data_np.reshape(1, -1)

prediction = logistic_regression_model.predict(input_data_reshaped)

if prediction == ['R']:
    print("This is Rock")
else:
    print("This is Mine")


