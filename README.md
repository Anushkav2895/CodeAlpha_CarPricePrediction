# CodeAlpha – Car Price Prediction with Machine Learning

Predicts the resale (selling) price of a used car from its features using regression models in Python (Pandas, Scikit-learn, Matplotlib, Seaborn).

## Structure
```
data/car_data.csv
car_price_prediction.py
requirements.txt
outputs/   analysis_output.txt, model_comparison.csv, best_model.joblib, figures/ (8 charts)
```

## Run
```
pip install -r requirements.txt
python car_price_prediction.py
```

## Dataset
301 used cars: `Car_Name, Year, Selling_Price, Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner`. Prices are in lakhs (INR 100,000). The data has no horsepower or brand-goodwill column, so the car model name is used as a brand/model signal instead.

## Workflow
1. **Cleaning:** checked missing values (none), removed 2 duplicate rows.
2. **Feature engineering:** `Car_Age`, `Kms_per_Year`, `Log_Driven_kms`, and `Model` (rare car names grouped as "other").
3. **Preprocessing:** scaling for numeric columns, one-hot encoding for categorical columns (inside a Scikit-learn Pipeline).
4. **Models:** Linear Regression, Ridge, Random Forest, Gradient Boosting (tree models trained on log price).
5. **Evaluation:** repeated 5-fold cross-validation on all data (main score) plus a held-out 20% test set; R², MAE, RMSE.

## Results
| Model | CV R² | CV MAE (lakhs) | Test R² |
|---|---|---|---|
| Gradient Boosting | 0.895 | 0.64 | 0.952 |
| Random Forest | 0.881 | 0.68 | 0.887 |
| Ridge | 0.800 | 1.21 | 0.800 |
| Linear Regression | 0.778 | 1.23 | 0.792 |

Best model: **Gradient Boosting** (average error about 0.6 lakhs). The test score is higher than the CV score because one 60-row split is noisy; the CV score is the more reliable estimate.

## Insights
- The showroom price (`Present_Price`) is by far the strongest predictor (~92% of feature importance), followed by car age.
- Diesel, automatic and dealer-sold cars tend to have higher prices.
- Mileage and owner count matter much less once price and age are known.

## Real-world use
Used-car marketplaces, dealers and lenders use similar models to suggest listing prices and estimate resale value.

## Limitations
Small dataset (299 rows after cleaning), a few very expensive cars, and no engine/horsepower data. Predictions for cars far outside the training range are less reliable.
