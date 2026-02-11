Bengaluru Real Estate Price Prediction

A machine learning project that predicts residential property prices in Bengaluru, India using features such as location, square footage, number of bedrooms, and bathrooms. The project walks through a full data science pipeline — from raw data cleaning through outlier removal, feature engineering, and model selection — ultimately identifying **Linear Regression** as the best-performing model with an **R² of ~0.83**.

---

Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation \& Setup](#installation--setup)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Key Libraries](#key-libraries)
- [License](#license)

---

Project Overview

The Bengaluru housing market is one of the most dynamic in India, with property prices varying dramatically by neighborhood, property size, and configuration. This project builds a predictive model for house prices (in Lakhs INR) by:

1. **Cleaning and preparing** a messy real-world dataset (handling mixed-format square footage values, missing data, and inconsistent size labels).
2. **Engineering features** such as price per square foot and consolidated location categories.
3. **Removing outliers** using multiple strategies — statistical thresholds on price per sqft, cross-BHK price anomaly detection, and bathroom count constraints.
4. **Comparing regression algorithms** via GridSearchCV with ShuffleSplit cross-validation to select the best model.

---

Dataset

The dataset (`Bengaluru_House_Data.csv`) contains **13,320 property listings** with the following original columns:

| Feature | Description |
|---|---|
| `area_type` | Type of area (Super built-up, Built-up, Plot, Carpet) |
| `availability` | When the property is available |
| `location` | Neighborhood / locality in Bengaluru |
| `size` | Number of bedrooms (e.g., "2 BHK", "4 Bedroom") |
| `society` | Name of the housing society |
| `total_sqft` | Total area in square feet (can contain ranges) |
| `bath` | Number of bathrooms |
| `balcony` | Number of balconies |
| `price` | Price in Lakhs INR (target variable) |

After cleaning, feature engineering, and outlier removal, the final modeling dataset contains **7,251 records**.

---

Methodology

1. Data Cleaning

- Dropped low-signal columns: `availability`, `society`, `balcony`, `area_type`.
- Removed rows with missing values (reduced from 13,320 to 13,246 records).
- Extracted numeric BHK count from the `size` column.
- Converted `total_sqft` from mixed formats (ranges like "1133 - 1384", text entries) into numeric values by averaging range endpoints.

2. Feature Engineering

- Created `price_per_sqft` for outlier analysis.
- Consolidated rare locations (≤10 listings) into an "other" category to reduce dimensionality.
- One-hot encoded the `location` feature for model input.

3. Outlier Removal (Multi-Stage)

- **Minimum sqft per BHK**: Removed properties with less than 300 sqft per bedroom (13,320 → 12,502).
- **Price per sqft by location**: Removed listings beyond ±1 standard deviation of the location-level mean price per sqft (12,502 → 10,241).
- **Cross-BHK anomaly detection**: Removed properties where a higher-BHK unit is priced lower per sqft than the mean of the lower-BHK category in the same location.
- **Bathroom constraint**: Removed properties where bathrooms exceed BHK count + 2 (final: 7,251 records).

4. Model Comparison

Three algorithms were compared using `GridSearchCV` with 5-fold `ShuffleSplit` cross-validation (test size = 33%):

- **Linear Regression** — with `fit_intercept` grid `[True, False]`
- **Decision Tree Regressor** — with `max_depth` grid `[1, 5, 10]`
- **Lasso Regression** — with `alpha` grid `[1, 2]` and `selection` grid `[random, cyclic]`

---

Results

Model Comparison (Cross-Validated R² Scores)

| Model | Best CV R² Score | Best Parameters |
|---|---|---|
| **Linear Regression** | **0.831** | `fit_intercept: False` |
| Decision Tree | 0.743 | `max_depth: 10` |
| Lasso | 0.704 | `alpha: 1, selection: random` |

Linear Regression — Detailed Evaluation

- **Test set R²**: 0.808
- **5-fold cross-validation scores**: 0.824, 0.797, 0.862, 0.824, 0.846

Linear Regression was selected as the best model given its superior and consistent performance across all folds.

---

Installation & Setup

Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run the Notebook

```bash
jupyter notebook Real_Estate_Model_Project__1_.ipynb
```

Ensure `Bengaluru_House_Data.csv` is in the same directory as the notebook.

---

Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Install the dependencies listed above.
3. Place `Bengaluru_House_Data.csv` in the project root.
4. Open and run `Real_Estate_Model_Project__1_.ipynb` sequentially.

---

Repository Structure

```
├── Real_Estate_Model_Project__1_.ipynb   # Main analysis notebook
├── Bengaluru_House_Data.csv              # Dataset (not included — see Dataset section)
├── README.md                             # This file
├── Report.pdf                            # Project report
```

---

Key Libraries

- **pandas / numpy** — Data manipulation and cleaning
- **matplotlib / seaborn** — Visualization and scatter plots
- **scikit-learn** — Train-test split, Linear Regression, Decision Tree, Lasso, GridSearchCV, cross-validation

---
