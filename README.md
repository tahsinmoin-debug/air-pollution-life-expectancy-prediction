# Air Pollution (PM2.5) & Life Expectancy Prediction

How much does air pollution matter for how long people live? This project merges the **WHO Life Expectancy** dataset with **World Bank PM2.5 air pollution** measurements and trains 10 regression models to predict life expectancy by country and year. Built as a group project for **CSE427 (Machine Learning), BRAC University**.

## Data

| Dataset | Source | In repo |
|---|---|---|
| Life Expectancy (WHO), 2000–2015, 193 countries | [Kaggle](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who) | Not included. Download `Life Expectancy Data.csv` from Kaggle |
| PM2.5 mean annual exposure (µg/m³), indicator `EN.ATM.PM25.MC.M3` | [World Bank](https://data.worldbank.org/indicator/EN.ATM.PM25.MC.M3) | [`data/raw/worldbank_pm25_by_country.csv`](data/raw/) |
| Merged dataset (164 countries, 2,519 rows) | Built in the notebook | [`data/merged_life_pm25.csv`](data/) |
| Final modelling dataset (2,500 rows × 17 cols) | Built in the notebook | [`data/final_life_pm25.csv`](data/) |

The two sources were joined on **country + year**, with the World Bank data reshaped from wide (one column per year) to long format.

## Workflow

1. **Merging:** reshaped the World Bank PM2.5 data and joined it with WHO life expectancy on country–year
2. **Cleaning:** standardised column names, removed invalid targets and rows missing country/year, median imputation, IQR outlier removal on life expectancy
3. **Feature engineering:** `population_log`, encoded `status` (Developed/Developing) and country
4. **Feature selection:** dropped features with |r| > 0.9 plus redundant columns (BMI, expenditure, thinness, income composition, etc.)
5. **EDA:** life expectancy distribution, PM2.5 vs life expectancy, correlation heatmap
6. **Modelling:** 80/20 split, 10 regressors, evaluated on train and test sets to check for overfitting

## Results (test set)

| Model | R² | RMSE (years) | MAE (years) |
|---|---|---|---|
| **XGBoost** | **0.963** | **1.79** | **1.07** |
| Extra Trees | 0.962 | 1.81 | 1.13 |
| Random Forest | 0.958 | 1.91 | 1.20 |
| Gradient Boosting | 0.957 | 1.92 | 1.26 |
| Decision Tree | 0.914 | 2.73 | 1.95 |
| kNN | 0.893 | 3.04 | 2.05 |
| MLP | 0.886 | 3.13 | 2.00 |
| AdaBoost | 0.876 | 3.27 | 2.54 |
| Ridge Regression | 0.816 | 3.99 | 2.99 |
| Linear Regression | 0.816 | 3.99 | 2.99 |

**Takeaways**
- Ensembles (XGBoost, Extra Trees, Random Forest) predict life expectancy to within about **1–2 years** on average.
- Linear models top out around R² 0.82, so the relationships are clearly non-linear.
- PM2.5 is negatively correlated with life expectancy (r ≈ −0.41). Schooling (+0.76), adult mortality (−0.69) and HIV/AIDS (−0.58) are correlated more strongly.
- kNN (train R² = 1.0) and XGBoost (train R² = 0.999) show a train/test gap, so they overfit to some degree.

## Repo structure

```
├── data/
│   ├── raw/worldbank_pm25_by_country.csv
│   ├── merged_life_pm25.csv
│   └── final_life_pm25.csv
├── notebooks/
│   └── pm25_life_expectancy_prediction.ipynb   # full pipeline with outputs
└── requirements.txt
```

## Run it

The notebook was written in Google Colab and reads from `/content/drive/MyDrive/`. To run it locally, `pip install -r requirements.txt` and change those paths to point at `data/`. To go straight to modelling, load `data/final_life_pm25.csv` and skip the merge cells.

## Next steps

- Use a time-based or country-grouped split. A random split lets the same country show up in both train and test, which inflates scores.
- Isolate PM2.5's effect with SHAP or partial dependence plots
- Add newer PM2.5 years and other pollutants (NO₂, O₃)
