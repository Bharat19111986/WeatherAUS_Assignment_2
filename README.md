# WeatherAUS Rainfall Classification

## a. Problem statement

Use Australian daily weather observations to predict whether it will rain tomorrow
(`RainTomorrow`: Yes/No). Compare five classification algorithms using one held-out
test set. The Streamlit app evaluates saved models; it never fits a model to uploaded
test data.

## b. Dataset description

The supplied WeatherAUS workbook contains 145,460 observations from 49 Australian
weather stations, 22 predictor columns, and the `RainTomorrow` target. After rows
without a target are removed, 142,193 labeled observations remain. The predictors
include numeric weather measurements and categorical station, wind, and rainfall
fields. Missing input values are imputed inside each fitted pipeline. `Date` is
converted to year, month, and day features. The target classes are `No` and `Yes`.

The workbook is retained at `data/weatherAUS.csv.xlsx` for reproducibility. The
training script reserves a stratified holdout and uses a stratified sample of up to
20,000 training rows and 5,000 test rows to keep all five models practical to train.
The resulting `test_data.csv` is the labeled sample used by the app and must include
all source predictor columns plus `RainTomorrow`.

## c. GitHub Repository Link

**Repository:** _Add the GitHub URL after publishing this local repository._

## d. Models used

All models share the same stratified training and test rows. Numeric columns are
median-imputed and standardized; categorical columns are imputed and one-hot encoded.
Precision, recall, and F1 use `Yes` as the positive class.
The Support Vector Machine uses a calibrated linear SVM so its probability scores
can be used to calculate AUC.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8062 | 0.8821 | 0.5466 | 0.7948 | 0.6478 | 0.5377 |
| Decision Tree | 0.7828 | 0.7937 | 0.5113 | 0.7074 | 0.5936 | 0.4616 |
| KNN | 0.8416 | 0.8567 | 0.7519 | 0.4380 | 0.5536 | 0.4904 |
| Naive Bayes (Gaussian) | 0.6606 | 0.7549 | 0.3691 | 0.7244 | 0.4890 | 0.3080 |
| Support Vector Machine | 0.8608 | 0.8820 | 0.7728 | 0.5370 | 0.6337 | 0.5651 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Tied for the strongest AUC (0.8821) and has the highest recall (0.7948), but its precision is lower (0.5466), so it raises more false alarms. |
| Decision Tree | Detects many rainy days (0.7074 recall) but has the lowest AUC (0.7937) and a lower MCC than the linear models. |
| KNN | Has the second-highest accuracy (0.8416) and good precision (0.7519), but its lowest recall (0.4380) means it misses many rainy days. |
| Naive Bayes | Its recall is relatively high (0.7244), but the lowest accuracy (0.6606) and precision (0.3691) reflect many false rain alerts. |
| Support Vector Machine | Leads accuracy (0.8608) and MCC (0.5651), with the highest precision (0.7728) and AUC close to Logistic Regression; recall is more moderate (0.5370). |
| Overall winner for this dataset? | Support Vector Machine is the strongest balanced choice by accuracy, MCC, and precision. If minimizing missed rainy days is the priority, Logistic Regression's higher recall is preferable. |

## Project files

```text
WeatherAUS_Assignment_2/
├── app.py
├── data/
│   └── weatherAUS.csv.xlsx
├── models/
│   ├── *.joblib
│   ├── metadata.json
│   └── metrics.json
├── requirements.txt
├── test_data.csv
├── train_models.py
└── weather_ml.py
```

## Run locally

1. Install Python and create an environment.
2. Install packages: `pip install -r requirements.txt`.
3. Train and save all five pipelines: `python train_models.py`.
4. Start the interface: `streamlit run app.py`.
5. Upload `test_data.csv` or another labeled WeatherAUS CSV in the app.

The test CSV must contain every source predictor column and `RainTomorrow`. The app
shows Accuracy, AUC, Precision, Recall, F1, and MCC for all five saved models on
the same uploaded test CSV, plus a confusion matrix and classification report for
the selected model.

