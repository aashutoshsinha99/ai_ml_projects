# California Housing Price Prediction

A machine learning project to predict house prices in California using XGBoost regression.

## Dataset
California housing dataset from scikit-learn with 8 features:
- MedInc, HouseAge, AveRooms, AveBedrms
- Population, AveOccup, Latitude, Longitude
- Target: Median house value (in $100,000 units)

## Process
1. Load and explore data
2. Correlation analysis with heatmap
3. Train-test split (80-20)
4. XGBoost Regressor model
5. Evaluate using R² score and MAE

## Results
- **R² Score**: 0.985 (98.5% of variance explained)
- **Mean Absolute Error**: $9,749
- **Average House Price**: $208,242

## Requirements
pandas, matplotlib, seaborn, scikit-learn, xgboost

## Usage
```bash
python house_price.py