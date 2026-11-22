# House Price Prediction with Decision Trees and Random Forests

A supervised machine learning project that predicts residential property sale prices using Python, Pandas, and Scikit-learn. Developed as a practical solution to Kaggle's Housing Prices Competition for Kaggle Learn Users, this project compares a baseline decision tree, a tuned decision tree, and a random forest regressor.

## Project overview

The objective is to estimate a home's **`SalePrice`** from seven numerical property characteristics. After evaluating the models using mean absolute error (MAE) on a held-out validation set, the random forest achieved the lowest MAE among the models tested.

The random forest's validation MAE was **$21,857**, compared with **$29,653** for the baseline decision tree and **$27,283** for the tuned decision tree.

## Dataset

- **`data/train.csv`** contains property attributes and the known sale price (`SalePrice`) used for model development.
- **`data/test.csv`** contains property attributes without the `SalePrice` target; predictions are generated for these records.

Only the following seven input features are used in this version:

| Feature | Description |
| --- | --- |
| `LotArea` | Lot size in square feet |
| `YearBuilt` | Original construction year |
| `1stFlrSF` | First-floor area in square feet |
| `2ndFlrSF` | Second-floor area in square feet |
| `FullBath` | Number of full bathrooms |
| `BedroomAbvGr` | Number of above-ground bedrooms |
| `TotRmsAbvGrd` | Total above-ground rooms (excluding bathrooms) |

**Prediction target:** `SalePrice` (USD).

## Methodology

### 1. Data preparation and validation split

The CSV files are loaded with Pandas. The seven selected features form the input matrix (`X`), and `SalePrice` forms the target (`y`). The labelled data is split into training and validation subsets using `train_test_split` with `random_state=1` and Scikit-learn's default 75% / 25% split.

### 2. Baseline decision tree

A `DecisionTreeRegressor(random_state=1)` is trained on the training subset and evaluated on the validation subset. Its MAE provides a baseline for comparison.

### 3. Decision tree tuning

The `max_leaf_nodes` parameter is evaluated using the following candidate values:

```python
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
```

The candidate with the lowest validation MAE is selected. In this experiment, the best candidate was **100 leaf nodes**. The selected decision tree is then fitted on the complete labelled training dataset before making test-set predictions.

### 4. Random forest

A `RandomForestRegressor(random_state=1)` is trained and evaluated using the same training/validation split. A separate random forest is then fitted on the complete labelled dataset to generate predictions for the unlabelled test data.

### 5. Submission generation

Predictions from the final random forest are saved to `submission.csv` using the format expected by the Kaggle competition:

```text
Id,SalePrice
1461,122656.58
1462,156789.00
1463,182959.00
...
```

These are **predictions**, not verified sale prices. The test data does not include target values, so its prediction error cannot be calculated locally from these files.

## Results

**Metric:** Mean Absolute Error (MAE), the average absolute difference between a model's predictions and actual sale prices. **Lower is better.**

| Model | Validation MAE | Notes |
| --- | ---: | --- |
| Decision Tree (default) | $29,653 | Baseline model |
| Decision Tree (tuned) | $27,283 | `max_leaf_nodes=100` |
| **Random Forest** | **$21,857** | Lowest MAE among models evaluated |

On this validation split, the random forest reduced MAE by approximately **26.3%** relative to the baseline decision tree. The values above are from one particular split and should not be interpreted as performance guarantees on unseen data.

## Running the project locally

**1. Download or clone the repository.** Place `train.csv` and `test.csv` inside a `data/` folder at the project root.

**2. Install dependencies** in your Python environment:

```bash
python -m pip install pandas scikit-learn notebook
```

If you have a `requirements.txt`, you can use `python -m pip install -r requirements.txt` instead.

**3. Start Jupyter Notebook:**

```bash
jupyter notebook
```

**4. Open and run `model.ipynb`.** The notebook reads `data/train.csv` and `data/test.csv`, fits the models, prints validation metrics and sample predictions, and writes `submission.csv` in its working directory.

Run Jupyter from the repository's root directory so these relative dataset paths resolve correctly.

## Tools and libraries

- **Python** — programming language
- **Pandas** — CSV loading and tabular data handling
- **Scikit-learn** — train/validation splitting, regression models, and error measurement
- **Jupyter Notebook** — interactive development and documentation

## Limitations and possible improvements

Its main limitations are:

- **Limited features:** Only seven numerical property attributes are considered; other relevant housing characteristics are excluded.
- **Single validation split:** MAE may change with another split. Cross-validation would provide a more robust estimate.
- **Tree tuning on the validation set:** The decision tree's leaf count is selected using the same validation set used to report its MAE. A separate test set with labels or nested validation would provide a more independent final assessment.
- **Limited hyperparameter search:** The random forest uses default hyperparameters apart from `random_state` and has not been systematically tuned.
- **No documented missing-value or categorical-variable pipeline:** Extending the feature set would require careful preprocessing.

Possible next steps include cross-validation, better feature selection, missing-value handling, encoding categorical features, random forest hyperparameter tuning, and examining feature importance.

## What I learned

This project was an opportunity to practise the core components of a regression pipeline:

- Separating input features from the prediction target
- Training and validating supervised learning models
- Understanding overfitting and controlling decision-tree complexity
- Comparing models using MAE rather than training performance alone
- Retraining a selected model on all available labelled data for final predictions
- Saving predictions in the format required for a Kaggle submission

## Acknowledgements

Built while learning from [Kaggle Learn: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning), using Kaggle's housing-price prediction data.

**Author:** [nan0techn0](https://github.com/nan0techn0)
