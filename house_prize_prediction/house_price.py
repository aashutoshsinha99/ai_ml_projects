#!/usr/bin/env python
# coding: utf-8

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
print(housing)


import pandas as pd

#load it as pandas dataframe
df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
df.head()


# Add the target column
df['price'] = housing.target
df.head()


# Check shape
df.shape


# Check stats data
df.describe()


# somewhere the data is in thousands and elsewhere in points
# we HAVE to standardize it


# Pre processing
# check for null values
df.isnull().sum()


# Check corelation

# Why Bother Checking?
# Find what matters: Maybe "Number of bathrooms" matters more than "Year built"
# Remove useless stuff: Don't waste time on features that don't affect price
# Spot weirdness: If "Rooms" and "Price" are negatively correlated (rooms up, price down) - something's wrong with your data!

# Bottom line: You're asking "Which features actually predict house prices?" Correlation helps answer that.
correlation_data = df.corr()
correlation_data


# Lets draw heatmap to vizualize the correlation_data

import  matplotlib.pyplot as plt
import seaborn as sns

# make plotting on big figure (optional)
plt.figure(figsize=(10, 7 )) 

#heatmap
sns.heatmap(
    correlation_data,   # the correlation data
    annot=True,         # show numbers in each square
    fmt='.2f',          # Shows: 0.69, -0.15, etc.
    annot_kws={
        'size': 7,
        'color': 'black',
        'fontweight': 'normal',
        'style': 'italic',
        'rotation': 45
    },                  # Annotation KeyWordS. Lets us customize the annotations 
    cmap='coolwarm',    # red=positive, blue=negative
    cbar=True,          # show colorbar
    center=0,           # set white at 0
    square=True,        # make squares, not rectangles
    linewidths=1.5,     # thin lines between squares
    cbar_kws={
        'shrink': 0.8,
        'label': 'Correlation',
        'orientation': 'vertical',
        'ticks': [-0.8, -0.4, 0, 0.4, 0.8]  # Custom tick marks
    }                   # Color Bar KeyWordS. Lets us customize the color bar
)


# Now some core changes to the graph like title, labels for axes,tick labels
plt.title('California Housing Correlation Heatmap', 
    fontsize=16, 
    fontweight='bold',
    pad=20
)
plt.xlabel('Features', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')

plt.xticks(rotation=25, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.tight_layout()
plt.show()


# Lets start the main thing now
# Split data into input and target

X = df.drop(columns='price') # or X = df.drop(['price'], axis=1)
Y = df['price']
print(X)
print(Y)


# Split data into training and test data
# import the lib 
from sklearn.model_selection import train_test_split

# No need to use stratify as the output doesnt have any classes
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# X_train has 166 inputs ,  Y train has 166 outputs
# X_test has 42 inputs , Y test has 42 outputs
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# Import XGBoost for high-performance gradient boosting regression

from xgboost import XGBRegressor

model = XGBRegressor()

# train the model with training data (X_train) and expected output(Y_train)
model.fit(X_train, Y_train)

#check accuracy
print(model.score(X_train, Y_train))


# Now do the same training on test data
model.fit(X_test, Y_test)

#check accuracy
print(model.score(X_test, Y_test))


# Now we can also check 
#  - R-Squared (R²) - "How good is my model?"
#  - Mean Absolute Error (MAE) - "How wrong am I?"

from sklearn.metrics import r2_score, mean_absolute_error

predicted_y_test = model.predict(X_test)
r2 = r2_score(predicted_y_test, Y_test)
mae = mean_absolute_error(predicted_y_test, Y_test)

print(f"R² Score: {r2:.3f}")
print(f"Mean Absolute Error: ${mae:.3f} (in 100k units)")


# Some print statements to show off
print(f"\nIn actual dollars:")
print(f"MAE: ${mae * 100000:,.0f}")
print(f"Average house price: ${Y_test.mean() * 100000:,.0f}")
print(f"R²: {r2:.1%} of variance explained")



Y_test.shape


# Wana visualize the actual output vs what the model predicted?
# Ok lets do it
plt.scatter(x=Y_test, y=predicted_y_test, s=1)
plt.title('Actual VS Predicted Prices', 
    fontweight='bold',
    pad=20
)
plt.xlabel('Actual Prices', fontsize=12, fontweight='normal')
plt.ylabel('Predicted Prices', fontsize=12, fontweight='normal')
plt.tight_layout()
plt.show()

