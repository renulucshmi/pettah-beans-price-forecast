# Pettah Beans Price Forecast

Forecasting **next week's average wholesale price of beans at Pettah market, Colombo**, with ensemble learning: bagging, boosting, stacking, voting and a Random Forest + ARIMA hybrid. Includes the full analysis notebook, the saved final model and a Streamlit dashboard.

**Author:** Prakasan R. (224152U), University of Moratuwa
**Module:** Machine Learning, ensemble learning assignment

---

## The problem

The wholesale price of beans at Pettah swings sharply from week to week (weekly averages from Rs 40/kg in April 2020 to about Rs 940/kg in January 2024). A one-week-ahead forecast is useful to traders, retailers and buyers planning purchases.

**Target:** next week's change in the Pettah beans price (as a log change), turned back into Rs/kg for reporting. A change is predicted instead of the price itself because tree models cannot predict values outside the range they were trained on, and prices after 2022 are much higher than before.

**Inputs (20 features):** recent price changes and volatility, the Dambulla market price, carrot and cabbage prices, weekly rainfall in Nuwara Eliya and Badulla (this week and the two weeks before), and the calendar (season, April New Year, December).

## Data (Sri Lankan sources only)

| Data | Source | File |
|---|---|---|
| Daily wholesale food prices, Aug 2016 to Nov 2026 (Pettah beans, Dambulla beans, Pettah carrot, Pettah cabbage) | **Central Bank of Sri Lanka, Daily Price Report**, compiled by Macro Lab: https://macrolab.lk/data/prices-and-inflation/daily-prices/ | `dataset/daily-prices-d.csv` |
| Daily rainfall for Nuwara Eliya (6.95° N, 80.79° E) and Badulla (6.99° N, 81.06° E) | **Open-Meteo Historical Weather API** (ERA5 reanalysis): https://open-meteo.com/ (downloaded by the notebook in Step 2) | `dataset/rain_daily.csv` |

The daily prices are averaged into **537 weeks** (weeks ending on Friday). 10 weeks with no reported trading days (1.9%) were filled by time interpolation and flagged in the `was_missing` column.

Files created by the notebook: `dataset/weekly_clean.csv`, `dataset/weekly_with_rain.csv`, `dataset/features.csv`, and the result tables in `results/`.

## Method

All models are trained on **Sep 2016 to Dec 2023** (383 weeks), tuned on **2024** (52 weeks, validation) and tested once on **Jan 2025 to Nov 2026** (97 weeks, test). The split is by time, so no model ever sees the future.

| Notebook step | What it does |
|---|---|
| 1 | Load the Macro Lab file, choose the columns, average into weeks, fill and flag missing weeks, data checks (duplicates, impossible prices, week spacing) |
| 2 | Download rainfall for the growing areas and check how rain lines up with price changes |
| 3 | Build the target and the 20 features, with automatic checks against data leakage |
| 4 | Baselines: naive "no change", seasonal naive, multiple linear regression, a single decision tree |
| 5 | Bagging: BaggingRegressor and Random Forest with out-of-bag (OOB) scores and feature importance |
| 6 | Boosting: Gradient Boosting, XGBoost and CatBoost, learning rate vs number of trees, tree depth, early stopping, a 10-seed check |
| 7 | Stacking and voting with five different base models (RF, GB, XGBoost, SVR, KNN) |
| 8 | Hyperparameter tuning with RandomizedSearchCV and time-series cross-validation, with the reason for each parameter |
| 9 | Hybrid model: ARIMA on the price history, plus a Random Forest that learns ARIMA's errors |
| 10 | Final comparison of all 20 models: table, charts, results by test year (2025 vs 2026), the Lecture 5 bagging / boosting / stacking table, Random Forest vs XGBoost, and interpretation |
| 11 | Re-train the final model on all weeks, save it with joblib, check that it reloads correctly; research note and conclusion |

LightGBM is not used: it crashed the Python kernel on the Windows laptop used for this project, so CatBoost was used as the third boosting library (the brief allows LightGBM or CatBoost).

## Results

Models are ranked by **unseen RMSE**: one error over all 149 weeks the models never trained on (2024 validation + 2025-26 test). RMSE and MAE are in Rs/kg; R² is for the weekly change.

| Model | Unseen RMSE | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|
| **Hybrid: RF + ARIMA (final model)** | **87.6** | **83.9** | **65.7** | **0.36** |
| Gradient Boosting | 89.5 | 85.5 | 67.3 | 0.35 |
| Voting (RF+GB+XGB+SVR+KNN) | 90.4 | 86.7 | 66.1 | 0.34 |
| Random Forest | 90.4 | 87.8 | 69.8 | 0.32 |
| Bagging (300 trees) | 90.7 | 87.6 | 69.7 | 0.32 |
| XGBoost | 91.8 | 89.0 | 68.7 | 0.34 |
| CatBoost | 92.8 | 90.8 | 68.4 | 0.31 |
| Stacking (RF+GB+XGB+SVR+KNN) | 95.5 | 91.5 | 68.0 | 0.30 |
| Linear regression | 99.4 | 97.2 | 65.6 | 0.33 |
| ARIMA(1,1,1) alone | 104.9 | 101.5 | 75.0 | 0.11 |
| Naive (no change) | 110.6 | 107.4 | 81.7 | 0.00 |
| Decision tree | 129.9 | 120.1 | 90.2 | -0.03 |

The full table of all 20 models, with training times and overfitting gaps, is in `results/results_final.csv`.

**Main findings**
- **Bagging fixed the single tree's overfitting** (test RMSE Rs 120 → Rs 88). **Boosting reached a similar level** (Gradient Boosting Rs 85.5); a 10-seed check showed the gap between them is within random variation. Weekly prices are mostly noise, so reducing variance (bagging) matters more than reducing bias (boosting).
- **The OOB score was too pessimistic** (OOB R² about 0.04 vs test R² 0.32), because the rain-price relationship is stronger after 2022 than in earlier years.
- **Rain in Nuwara Eliya this week** is the strongest predictor in every model.
- **Voting beat stacking**: the three tree models make almost the same predictions (correlation 0.93-0.97), so learning weights mostly fitted noise.
- **Tuning improved the cross-validation score but made the test score worse**, because the CV folds come from years where the signal was weaker.
- **The RF + ARIMA hybrid is best**: ARIMA uses only past prices, and the forest adds what ARIMA cannot see (rain, the other market, the season). It beats both of its parts.
- **Limits:** the best model is still off by about Rs 84/kg on a typical week and is often a week late on sudden jumps. It gives the direction and rough level, not an exact price.

## Bagging, boosting, stacking, voting and the hybrid

| | Bagging / Random Forest | Boosting | Stacking | Voting | Hybrid (RF + ARIMA) |
|---|---|---|---|---|---|
| What is combined | many full-depth trees of the same kind | many small trees of the same kind | different kinds of models (RF, GB, XGBoost, SVR, KNN) | the same five models | a time-series model and a tree ensemble |
| How they are trained | independently, each on a bootstrap sample | one after another, each fitting the errors left so far | base models first, then a meta-model on their out-of-fold predictions | independently | ARIMA first, then the forest learns ARIMA's errors |
| How predictions are combined | plain average | weighted sum of all trees | weights learned by the meta-model | plain average | ARIMA forecast + forest correction |
| Mainly reduces | variance (overfitting) | bias (underfitting) | both, if the base models make different errors | variance | errors that one model type cannot see |
| Main risk here | cannot predict outside past values (handled by predicting the change) | overfits with a large learning rate or deep trees (default XGBoost) | the meta-model overfits noisy, older data | none specific; can be slightly worse than the best single model | more parts to save and keep consistent |
| Result (test RMSE) | Rs 87.6-87.8 | Rs 85.5 (Gradient Boosting) | Rs 91.5 | Rs 86.7 | **Rs 83.9** |

## Methodology checks

| Check | How it is handled |
|---|---|
| No future information in the features | Every feature uses only the current week and earlier weeks; the calendar features use next week's date, which is known in advance. Step 3 asserts that the target is exactly next week's change, that no next-week column is an input, and that train, validation and test do not overlap. |
| Time-based split | Train ≤ 2023 < validation 2024 < test 2025-26; no shuffling anywhere. |
| Settings chosen without the test set | Learning rate, tree depth, early stopping, SVR/KNN settings and the stacking setup were chosen on the training data or the 2024 validation year. |
| ARIMA does not look ahead | Its parameters are learned on 2016-2023 and frozen; every later forecast uses only the weeks before it. |
| App uses the same features as the notebook | `pipeline.py` is shared, and Step 11 asserts that it rebuilds exactly the Step 3 features and that the saved model reproduces the notebook's predictions. |
| Reproducibility | `random_state = 42` everywhere; a 10-seed check (Step 6) shows how much scores move with randomness alone. |

## Limitations

- **The final model was chosen using the test weeks too.** The ranking uses validation and test together (Step 10) to avoid relying on one year, so the hybrid's test score is slightly optimistic. The top five models are within about Rs 3 of each other, so the exact order is not certain.
- **Stacking uses `KFold`, not a walk-forward split.** For some training weeks, the base models that produced the meta-model's training predictions had seen later weeks. The final test is still strictly on 2025-26.
- **One test period** (2025 to Nov 2026) and no statistical significance test of the differences between models.
- **Rainfall is from a reanalysis grid** (ERA5 via Open-Meteo), not from weather-station readings in the growing areas.
- **Prices are compiled by Macro Lab** from the Central Bank's Daily Price Report; Macro Lab notes that its files are not an official publication of the Central Bank.
- **Missing drivers:** fuel and transport costs, imports, festival demand and road closures are not in the data, so sudden jumps are often forecast a week late.

## Research note: Random Forest vs gradient boosting, tabular vs image and text data

**Tabular data** (like this project's prices and rainfall): tree ensembles are usually the strongest choice. They need no scaling, handle features in different units, find non-linear effects and interactions on their own, and work with small datasets. **Random Forest** trains independent trees in parallel, gives an OOB error and feature importance for free, and needs little tuning. **Gradient boosting** (Gradient Boosting, XGBoost, CatBoost) trains trees one after another and can be more accurate when there is a lot of signal, but it is more sensitive to its settings and can overfit noise. Both share one limitation: trees cannot predict values outside the range seen in training, which is why this project predicts the weekly change and adds ARIMA for the price history.

**Image and text data:** the meaning is in the structure (neighbouring pixels form shapes; word order gives meaning). A tree looks at one input value at a time, so it cannot learn these patterns from raw pixels or raw text, and the number of inputs is very large. Convolutional networks and transformers are built for this and are the standard choice. Tree ensembles are still useful once images or text have been turned into a table of features, such as word counts (TF-IDF) or features from a pre-trained network.

**In this project:** Random Forest (test RMSE Rs 87.8) and XGBoost (Rs 89.0) were effectively tied; Random Forest needed no tuning, while XGBoost needed a small learning rate, shallow trees and early stopping, and overfitted with its default settings (Rs 95.4).

## References

- Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123-140.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.
- Chen, T., & Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *Proceedings of KDD 2016*, 785-794.
- Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems 31*.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241-259.
- Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. *Neurocomputing*, 50, 159-175.
- Smith, J., & Wallis, K. F. (2009). A simple explanation of the forecast combination puzzle. *Oxford Bulletin of Economics and Statistics*, 71(3), 331-355.
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999-2049.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436-444.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 2022 Datasets and Benchmarks Track*.

## How to run

The project was built with **Python 3.11** (Anaconda). The saved model must be loaded with the same scikit-learn version it was saved with (1.3.0), so the versions are pinned in `requirements.txt`.

```bash
pip install -r requirements.txt
```

**Streamlit version:** `requirements.txt` pins `streamlit==1.60.0` on purpose. Streamlit 1.66 and newer require numpy 1.25+, and upgrading numpy breaks scikit-learn 1.3.0 and pandas 2.0.3. On Windows, close VS Code / Jupyter before installing, so no running kernel locks the package files.

### Notebook

Open `ensemble_assignment.ipynb` (VS Code or Jupyter) and run **Restart → Run All**. A full run takes about 10 minutes. Step 2 downloads rainfall from Open-Meteo, so it needs an internet connection.

### Streamlit dashboard

```bash
streamlit run app.py
```

It opens at http://localhost:8501 and has four tabs:
1. **Data overview:** latest prices, weekly Pettah and Dambulla prices, rainfall in the growing areas, the data sources.
2. **Model comparison:** the table of all 20 models (filter by family), charts of test error, training vs test error and training time, and actual vs predicted prices on the 2025-26 test weeks.
3. **What drives the price:** feature importance of the final model.
4. **Forecast:** next week's forecast (hybrid and ARIMA alone) with its typical error, and a "what if" form to change this week's prices and rainfall.

The app only needs `app.py`, `pipeline.py`, `model/hybrid_rf_arima.joblib`, `dataset/weekly_with_rain.csv`, `results/results_final.csv` and `results/test_predictions.csv`; it does not re-run the notebook.

## Project structure

```
├── ensemble_assignment.ipynb   analysis notebook (Steps 1-11, with explanations)
├── app.py                      Streamlit dashboard
├── pipeline.py                 feature building + forecast function, shared by the notebook and the app
├── requirements.txt            library versions
├── .streamlit/config.toml      light theme for the dashboard
├── model/
│   └── hybrid_rf_arima.joblib  saved final model (Random Forest + ARIMA)
├── dataset/
│   ├── daily-prices-d.csv      raw daily prices (Macro Lab / CBSL)
│   ├── rain_daily.csv          raw daily rainfall (Open-Meteo)
│   ├── weekly_clean.csv        weekly prices after cleaning
│   ├── weekly_with_rain.csv    weekly prices + weekly rainfall
│   └── features.csv            target and the 20 features
└── results/                    result tables for each step, the final comparison (results_final.csv)
                                and the test-period predictions (test_predictions.csv)
```

## AI tools used

**Tool:** Claude (Anthropic), used through the Claude app, throughout this project.

**What Claude did:**
- **Dataset search:** searched for Sri Lankan datasets suitable for the assignment and made a ranked comparison; I chose the CBSL / Macro Lab daily price data from that list.
- **Code:** wrote most of the Python code in the notebook (Steps 1-11), `pipeline.py` and `app.py`, and test-ran the notebook and the app to find errors before I ran them.
- **Explanations:** wrote the markdown explanations in the notebook, including the "why" for each choice and the "What this shows" notes, which were written from the outputs of my own runs.
- **Debugging:** helped fix problems on my laptop, for example the LightGBM kernel crash, the exact collinearity that broke linear regression (solved by leaving the duplicated feature `gap_ma4` out of that model), and the Streamlit / numpy version conflict.
- **Documentation:** drafted this README.

**What I did:**
- Chose the problem, the dataset and the target from the options Claude found.
- Ran every step of the notebook and the app on my own laptop, checked the outputs and charts, and asked for changes where something was wrong or unclear.
- Committed each step to this GitHub repository.

All numbers, charts and results in the notebook and this README come from my own runs of the code.
