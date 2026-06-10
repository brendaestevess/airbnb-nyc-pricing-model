# Airbnb NYC Pricing Model — Full Project Summary
### BANA 620 | Predictive Analytics & Data Mining

---

## What We Built

A **suggested nightly price tool** for new Airbnb hosts in NYC.
Given a listing's borough, room type, and activity metrics, the model predicts a reasonable price.

**Data:** 48,895 NYC Airbnb listings from 2019 (16 columns)
**Target:** `log(price)` — the natural log of nightly price
**Model:** Linear Regression inside a scikit-learn Pipeline

---

## SECTION A.1 — What the Data Looks Like

### Shape & Types
- **48,895 rows, 16 columns**
- Key numeric columns: `price`, `minimum_nights`, `number_of_reviews`, `availability_365`
- Key text columns: `neighbourhood_group`, `neighbourhood`, `room_type`

### Missing Values
| Column | Missing | What to do |
|---|---|---|
| `name` | 16 | Ignore — not used as a feature |
| `host_name` | 21 | Ignore — not used as a feature |
| `last_review` | 10,052 | Ignore — not a feature |
| `reviews_per_month` | 10,052 | Fill with **0** (explained below) |

### Price Distribution
- **Strongly right-skewed** — most listings are under $200/night, but a few go up to $10,000
- Mean = $153, Median = $106
- The mean is pulled up by expensive outliers; the median is not

### Median Price by Borough
| Borough | Median Price |
|---|---|
| Manhattan | $150 |
| Brooklyn | $90 |
| Queens | $75 |
| Staten Island | $75 |
| Bronx | $65 |

### Median Price by Room Type
| Room Type | Median Price |
|---|---|
| Entire home/apt | $160 |
| Private room | $70 |
| Shared room | $45 |

### Overall Median: $106/night

### Top 5 Neighbourhoods by Number of Listings
1. Williamsburg — 3,920
2. Bedford-Stuyvesant — 3,714
3. Harlem — 2,658
4. Bushwick — 2,465
5. Upper West Side — 1,971

---

## SECTION A.2 — Data Preprocessing (Cleaning the Data)

### Step 1: Drop price == 0
- **11 rows** removed
- Zero-price listings are inactive placeholders, not real rental offers

### Step 2: Cap price at $500
- **1,044 rows** removed (only 2.1% of the data)
- The 95th percentile is $355; the 99th is $799 — so $500 is a reasonable, round cut
- Removes luxury outliers that would distort the model for typical hosts
- **Too low a cap** (e.g. $200): throws away valid data; model understimates premium listings
- **Too high a cap** (e.g. $5,000): keeps extreme outliers that pull the model off track

### Step 3: Fill missing reviews_per_month with 0
- 10,052 listings have never received a review — they are new or inactive
- Their true review rate really is zero. Filling with 0 is correct, not a guess
- Using the mean/median would be wrong — it would imply they get average reviews, which is false

### Step 4: Create log_price = np.log(price)
- Raw price is skewed; `log(price)` is roughly bell-shaped (normal)
- Linear regression assumes normally distributed residuals — log satisfies this
- `np.log` is safe here because all `price == 0` rows are already removed
- Coefficients now represent **percentage effects**, which is intuitive for pricing

### Step 5: Define Features
| Type | Features |
|---|---|
| Categorical | `neighbourhood_group`, `room_type` |
| Numeric | `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365` |

> **Why not use `neighbourhood` (221 unique values)?**
> One-hot encoding 221 categories creates 220 new columns — most nearly empty.
> The model would overfit to noise. `neighbourhood_group` (5 boroughs) is clean and generalizable.

### Step 6: Train/Test Split
- **80% training** (38,272 rows), **20% test** (9,568 rows)
- `random_state=42` — ensures the same split every time (reproducible)

---

## SECTION A.3 — The Model

### How the Pipeline Works

```
Raw data
   → ColumnTransformer
       → OneHotEncoder   (neighbourhood_group, room_type → binary columns)
       → StandardScaler  (numeric features → mean=0, std=1)
   → LinearRegression
```

**Why use a Pipeline?**
The Pipeline fits the scaler on training data only and applies it to both train and test.
This prevents **data leakage** — if you scaled the full dataset first, the model would have
already "seen" the test set before evaluation, making results look better than they really are.

### Why scale before splitting is wrong (data leakage explained simply)
Imagine your exam answers are graded on a curve based on the class average.
If the grader sees your exam before computing the average, your score is unfairly influenced.
Scaling after splitting does the same thing — test data "leaks" into training statistics.

---

## SECTION A.3 — Model Results

### Performance Metrics
| Metric | Value | What it means |
|---|---|---|
| Test R² | **0.5283** | The model explains 53% of price variation |
| RMSE (log scale) | **0.4314** | Average error in log-price units |
| RMSE (dollars) | **$69.13** | Predictions are off by ~$69 on average |

### Top 5 Coefficients by Magnitude
| Feature | Coefficient | Meaning |
|---|---|---|
| `room_type_Entire home/apt` | +0.626 | Biggest driver — entire homes are ~87% more expensive |
| `room_type_Shared room` | −0.507 | Shared rooms are ~40% cheaper |
| `neighbourhood_group_Manhattan` | +0.360 | Manhattan adds ~43% premium |
| `neighbourhood_group_Staten Island` | −0.196 | Staten Island is ~18% cheaper |
| `neighbourhood_group_Bronx` | −0.177 | Bronx is ~16% cheaper |

---

## CONCEPTUAL QUESTIONS — Plain-English Answers

---

### Q: Why is median better than mean for summarizing price?

Price is right-skewed — a few $5,000+/night listings pull the mean upward.
The **mean ($153)** is inflated by those extremes.
The **median ($106)** is the true middle value and can't be moved by outliers.
For a "typical host" tool, the median is the honest benchmark.

---

### Q: What does a right-skewed price distribution mean for linear regression?

Linear regression assumes errors (residuals) are normally distributed and consistent in size.
A skewed target breaks both assumptions:
- **Heteroscedasticity:** errors grow larger for expensive listings
- **Outlier leverage:** a handful of $10,000 listings drag the whole regression line

Solution: model `log(price)` instead — compresses the tail, stabilizes errors.

---

### Q: Median $150 in Manhattan vs. $65 in Bronx — what does this mean for the tool?

**Recommendation:** Anchor the suggested price to the host's specific borough, not a citywide average.
Showing a Manhattan host $65 (the Bronx median) would cause them to leave money on the table.
Borough should be the first filter before any other adjustment.

---

### Q: 221 unique neighbourhoods vs. 5 boroughs — which to one-hot encode?

**Encode `neighbourhood_group` (5 boroughs → 4 dummy columns).**
Encoding `neighbourhood` (221 values → 220 dummy columns) causes:
1. **Sparsity/overfitting** — most columns are almost always 0
2. **Unseen categories** — new listings in rare neighbourhoods get no useful prediction

---

### Q: Why split train/test BEFORE fitting the scaler?

If you scale the full dataset first, the scaler learns the test set's statistics.
This is **test-set leakage** — the model has indirectly seen test data before evaluation.
Result: R² looks better than it really is on new data.
A Pipeline prevents this automatically — scaler is fit on `X_train` only.

---

### Q: Is R² = 0.53 a failure?

**No — it's a solid first version given the feature set.**

What the model CAN see: borough, room type, basic activity metrics.
What the model CANNOT see: apartment size, floor, views, amenities, photos, proximity to subway.

Two listings in the same borough with the same room type can differ by $100+ based on things
not in this dataset. R² = 0.53 is genuinely useful — it's far better than guessing the median
for everyone. The right framing: the model sets a defensible starting price; hosts adjust up or
down based on their unit's specific qualities.

---

### Q: Two features not in the dataset that would improve R²

**1. Number of bedrooms / capacity (how many guests it sleeps)**
Right now a studio and a 3-bedroom are both "Entire home/apt" with the same prediction.
Capacity is one of the strongest price drivers on any rental platform.

**2. Distance to nearest subway station**
NYC prices are tightly tied to transit access. A unit 2 blocks from the subway
commands more than one 20 minutes away, even in the same neighbourhood.
`neighbourhood_group` (5 values) is too coarse to capture this.

---

### Q: Interpreting the Entire home/apt coefficient (+0.626) for a non-technical PM

The model predicts log(price), so coefficients are multipliers, not dollar amounts.

> e^0.626 ≈ 1.87

**In plain English:** Listing an entire home/apt is associated with a price **~87% higher**
than a private room in the same borough — everything else equal.
If the baseline private room rents for $80/night, the model expects a comparable entire home
to fetch around $150.

---

### Q: A host gets a suggested price of $90 but a neighbor charges $150 — should they trust the tool?

**The tool is a starting point, not the final answer.**

The $90 prediction is an average across all listings with the same borough + room type profile.
It cannot see:
- Apartment size, floor, or views
- Renovated kitchen or premium amenities
- The host's photos and reputation
- Building features (doorman, gym, rooftop)

The $150 next door likely reflects one or more of those factors.

**Practical advice:** Start at $90 to attract your first bookings and reviews.
Once you have social proof and understand your listing's strengths, raise toward $150.
Your unit's specific qualities — not the model — justify the premium.

---

## Summary for the Product Team

| What the model does well | What the model misses |
|---|---|
| Sets borough- and room-type-aware baselines | Apartment size and capacity |
| Prevents wild over/under-pricing | Amenities (gym, doorman, WiFi) |
| Works for brand-new hosts with no history | Transit proximity and walkability |
| Fast, interpretable, no black box | Host reputation and photo quality |

**Bottom line:** R² = 0.53, RMSE ≈ $69. Room type and borough drive price far more than
any activity metric. This tool is ready for a beta launch as a "starting price suggester" —
with a clear message to hosts that their specific unit may warrant adjustment.
