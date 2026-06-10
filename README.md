# Airbnb NYC Pricing Model

A linear regression model that predicts nightly Airbnb prices in New York City. Built to help new hosts set a competitive starting price based on borough, room type, and listing activity.

**Course:** BANA 320 — Predictive Analytics & Data Mining  
**School:** California State University, Northridge  
**Author:** Brenda Esteves

---

## The Business Problem

New Airbnb hosts in NYC have no baseline for pricing their listing. Set it too high and you get no bookings. Set it too low and you leave money on the table. This model gives first-time hosts a data-driven starting point.

---

## Dataset

- **Source:** NYC Airbnb Open Data (2019)
- **Size:** 48,895 listings, 16 columns
- **Key columns:** `neighbourhood_group`, `room_type`, `minimum_nights`, `number_of_reviews`, `availability_365`, `price`

---

## What I Did

**Data Cleaning**
- Removed 11 listings with a price of $0 (inactive placeholders)
- Capped prices at $500 to remove luxury outliers (2.1% of data)
- Filled 10,052 missing `reviews_per_month` values with 0 — these listings genuinely had no reviews, so 0 is accurate, not an assumption

**Feature Engineering**
- Created `log_price` as the target variable to correct right skew and stabilize regression errors
- Used `neighbourhood_group` (5 boroughs) instead of `neighbourhood` (221 unique values) to avoid overfitting

**Model**
- Built a scikit-learn Pipeline with a ColumnTransformer
- OneHotEncoder for categorical features (borough, room type)
- StandardScaler for numeric features
- LinearRegression as the final estimator
- 80/20 train/test split with `random_state=42`

---

## Results

| Metric | Value |
|---|---|
| Test R² | 0.5283 |
| RMSE (log scale) | 0.4314 |
| RMSE (dollars) | ~$69 |

**Top 5 coefficients by magnitude:**

| Feature | Coefficient | Meaning |
|---|---|---|
| Entire home/apt | +0.626 | ~87% price premium over private room |
| Shared room | −0.507 | ~40% cheaper than private room |
| Manhattan | +0.360 | ~43% premium over other boroughs |
| Staten Island | −0.196 | ~18% cheaper than baseline |
| Bronx | −0.177 | ~16% cheaper than baseline |

Room type and borough are by far the strongest pricing signals. Activity metrics (reviews, availability) contribute very little.

---

## Key Takeaway

R² = 0.53 is a solid result given the available features. The model cannot see apartment size, amenities, transit proximity, or photos — all of which influence price significantly. It works best as a starting point for new hosts, not a final answer.

---

## Tools

- Python 3
- pandas, NumPy
- scikit-learn (Pipeline, ColumnTransformer, LinearRegression)
- matplotlib, seaborn
