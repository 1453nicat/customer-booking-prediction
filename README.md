# customer-booking-prediction
✈️ | Predicting customer booking completion using Random Forest - British Airways Data Science Task -.

<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/British_Airways_Logo.svg/320px-British_Airways_Logo.svg.png" width="180" alt="British Airways Logo" />

# ✈️ British Airways — Data Science Job Simulation

[![Forage](https://img.shields.io/badge/Forage-British%20Airways-00428a?style=for-the-badge)](https://www.theforage.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Completed%20✓-2ecc71?style=for-the-badge)]()
[![Certificate]([https://img.shields.io/badge/Certificate-March%202026-00428a?style=for-the-badge](https://www.theforage.com/completion-certificates/tMjbs76F526fF5v3G/NjynCWzGSaWXQCxSX_tMjbs76F526fF5v3G_699b43f63b2e4c13b63d62c9_1774201344641_completion_certificate.pdf))]()

> **Completed as part of the British Airways Data Science Job Simulation on Forage** — March 22, 2026
> Analysed customer review data and built a machine learning model to predict booking completion behaviour.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Tasks](#-tasks)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Feature Engineering](#-feature-engineering)
- [Model & Evaluation](#-model--evaluation)
- [Key Findings](#-key-findings)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Certificate](#-certificate)

---

## 🔍 Overview

This project is part of the **British Airways Data Science Job Simulation** hosted on [Forage](https://www.theforage.com/). The simulation replicates the real-world data science workflows used at one of the world's leading airlines — from scraping and analysing customer sentiment data to building a predictive model that identifies which customers are most likely to complete a booking.

The core business problem: **British Airways needs to understand what drives a customer to complete a booking**, enabling proactive, data-driven marketing and personalisation strategies before a customer even considers a competitor.

---

## 🗂️ Tasks

### Task 1 — Web Scraping & Sentiment Analysis
- Scraped customer reviews from [Skytrax](https://www.airlinequality.com/airline-reviews/british-airways)
- Cleaned and pre-processed unstructured text data
- Performed sentiment analysis to uncover key themes in customer feedback
- Summarised insights in an executive-ready PowerPoint slide

### Task 2 — Predictive Modelling of Customer Bookings *(this repo)*
- Explored and prepared the `customer_booking.csv` dataset (50,000 records)
- Engineered 5 new features to enhance predictive power
- Trained a **Random Forest Classifier** to predict booking completion
- Evaluated the model using **5-Fold Stratified Cross-Validation**
- Visualised feature importances and summarised findings in a single business slide

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | British Airways (via Forage simulation) |
| Records | 50,000 customer booking records |
| Original features | 14 |
| Target variable | `booking_complete` (binary: 0 / 1) |
| Class distribution | **85% not booked / 15% booked** (imbalanced) |

**Original columns:**

| Column | Description |
|---|---|
| `num_passengers` | Number of passengers travelling |
| `sales_channel` | Booking channel (Internet / Mobile) |
| `trip_type` | Round Trip, One Way, or Circle Trip |
| `purchase_lead` | Days between booking date and travel date |
| `length_of_stay` | Number of days at destination |
| `flight_hour` | Departure hour |
| `flight_day` | Day of week of departure |
| `route` | Origin → destination route code |
| `booking_origin` | Country the booking was made from |
| `wants_extra_baggage` | Whether customer selected extra baggage |
| `wants_preferred_seat` | Whether customer selected a preferred seat |
| `wants_in_flight_meals` | Whether customer selected in-flight meals |
| `flight_duration` | Total flight duration in hours |
| `booking_complete` | **Target** — 1 if booking was completed |

---

## ⚙️ Methodology
```
Raw Data → EDA → Feature Engineering → Encoding → Model Training → CV Evaluation → Interpretation
```

1. **Exploratory Data Analysis** — distribution plots by target, categorical booking rates, correlation heatmap
2. **Data Cleaning** — zero null values confirmed; `flight_day` mapped from string to numeric ordinal
3. **Feature Engineering** — 5 new features constructed from domain logic
4. **Encoding** — `LabelEncoder` applied to all categorical columns
5. **Modelling** — `RandomForestClassifier` with class balancing
6. **Evaluation** — 5-Fold Stratified Cross-Validation across 5 metrics
7. **Interpretation** — Feature importance ranking and business recommendation

---

## 🛠️ Feature Engineering

5 new features were constructed to give the model richer signals:

| New Feature | Logic | Rationale |
|---|---|---|
| `purchase_urgency` | Bins `purchase_lead` into 5 urgency tiers | Captures non-linear booking intent patterns |
| `total_extras` | Sum of 3 extras flags | Proxy for customer commitment and intent |
| `is_long_haul` | `flight_duration > 6` hours | Long-haul vs short-haul behaviour differs |
| `is_weekend_flight` | `flight_day` ∈ {Fri, Sat, Sun} | Weekend travellers book differently |
| `is_business_hours` | `flight_hour` between 8–18 | Business-hours bookings signal trip purpose |

> `total_extras` ranked **7th out of 19 total features** — validating that customer intent signals captured through add-on selection genuinely contribute predictive power.

---

## 🤖 Model & Evaluation

### Model Configuration
```python
RandomForestClassifier(
    n_estimators     = 200,        # 200 decision trees
    max_depth        = 10,         # Regularisation via depth limit
    min_samples_leaf = 10,         # Minimum 10 samples per leaf
    class_weight     = "balanced", # Compensates for 85/15 class imbalance
    random_state     = 42,
    n_jobs           = -1
)
```

### 5-Fold Stratified Cross-Validation Results

| Metric | Train Mean | Val Mean | Val Std | Status |
|---|---|---|---|---|
| Accuracy | 0.7325 | **0.7059** | ±0.0072 | ✅ Normal |
| F1 Score | 0.4700 | **0.4143** | ±0.0056 | ✅ Normal |
| ROC AUC | 0.8298 | **0.7640** | ±0.0075 | ✅ Normal |
| Precision | 0.3340 | **0.2951** | ±0.0055 | ✅ Normal |
| Recall | 0.7932 | **0.6951** | ±0.0104 | ✅ Normal |

> No overfitting detected — all Train vs Validation gaps are within the acceptable threshold of < 0.10.

### Top 10 Feature Importances

| Rank | Feature | Importance |
|---|---|---|
| 🥇 1 | `booking_origin` | 0.3617 |
| 🥈 2 | `route` | 0.1307 |
| 🥉 3 | `flight_duration` | 0.1050 |
| 4 | `length_of_stay` | 0.1025 |
| 5 | `purchase_lead` | 0.0651 |
| 6 | `flight_hour` | 0.0397 |
| 7 | `total_extras` *(engineered)* | 0.0352 |
| 8 | `wants_extra_baggage` | 0.0243 |
| 9 | `flight_day_num` | 0.0202 |
| 10 | `flight_day` | 0.0197 |

---

## 💡 Key Findings

- **`booking_origin` is the dominant predictor** (0.3617) — where a customer books from has the strongest influence on completion. This reflects cultural booking patterns, regional payment infrastructure, and BA route availability per market.

- **Flight characteristics outrank customer preferences** — `flight_duration`, `length_of_stay`, and `route` collectively outrank individual extras flags, suggesting trip structure is a stronger intent signal than add-on selection alone.

- **Engineered `total_extras` ranked 7th of 19 features** — confirms feature engineering added genuine value. Customers selecting multiple extras show meaningfully higher booking completion.

- **Internet channel converts at 15.5% vs Mobile at 10.8%** — channel type has a measurable impact on completion rate, pointing to UX or demographic differences between platforms.

- **ROC AUC of 0.764** indicates solid discriminative power for a real-world imbalanced problem. Enriching with features like loyalty tier, session duration, or device type could push this further.

---

## 📁 Project Structure
```
british-airways-data-science/
│
├── 📓 Getting_Started.ipynb      # Full analysis notebook with outputs
├── 📊 customer_booking.csv       # Dataset (50,000 records)
├── 📄 BA_Results_Slide.pptx      # Executive summary slide
│
├── outputs/
│   ├── feature_importance.png    # Feature importance bar chart
│   └── confusion_matrix.png      # Confusion matrix (test set)
│
└── README.md
```

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Wrangling | `pandas`, `numpy` |
| Machine Learning | `scikit-learn` |
| Visualisation | `matplotlib`, `seaborn` |
| Environment | Jupyter Notebook / Google Colab |

---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/1453nicat/british-airways-data-science.git
cd british-airways-data-science
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**3. Launch the notebook**
```bash
jupyter notebook Getting_Started.ipynb
```

> **Google Colab users:** Upload `customer_booking.csv` to `/content/` and run all cells.

---

## 🏆 Certificate

**British Airways Data Science Job Simulation** — Forage  
📅 Completed: **March 22, 2026**

**Skills demonstrated:**
`Data Science` · `Machine Learning` · `Data Modeling` · `Data Visualization` · `Communication` · `PowerPoint` · `Assumption Development` · `Infrastructure Planning`

---

<div align="center">

**Built as part of the British Airways × Forage Job Simulation**

[![GitHub](https://img.shields.io/badge/More%20Projects-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/1453nicat)

</div>
