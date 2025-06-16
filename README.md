# AI Impact on Business: Profit-Uplift Classifier

## 🚀 Project Overview
I built a model to predict whether a company will see a profit boost after adopting AI. Using financials, sector info, and a “customer-focused” flag, I trained an XGBoost classifier and regressor to answer two questions:
1. **Classification:** Will profit go up or not?  
2. **Regression:** By roughly how much (in USD)?  

This helps decision-makers flag which firms are ripe for AI investment, and by how much they might gain.

---

## 📊 Data Description
- **Source:** Cleaned CSV from Phase 2 (`final_dataset_phase2.csv`)  
- **Records:** ~ 5 000 companies with no missing `Profit Change`  
- **Key columns:**
  - `Profit 2022`, `Profit 2024` (numeric)  
  - `Sector` (categorical, encoded to `Sector_enc`)  
  - `Customer_Focused` (0/1)  
  - `Profit Change` = 2024−2022, our regression target  
  - `Target` = 1 if `Profit Change`>0, else 0 (classification label)  

---

## 🔍 Exploratory Analysis
- **Profit distributions:** Right-skewed; outliers trimmed/flagged.  
- **Sector vs. Profit Change:** Tech & Healthcare tended to see bigger gains.  
- **Customer-Focus:** Companies with customer-facing AI generally had higher profit upticks.  
- **Correlation heatmap:** Removed collinear pairs (e.g. raw 2022 vs. 2024 profits).

---

## 🛠 Feature Engineering
- **Sector encoding:** LabelEncoder → `Sector_enc`  
- **Binary target:** `Target = (Profit Change > 0)`  
- **Clean-up:** Dropped any rows with NaN or infinite `Profit Change`

---

## 🤖 Modeling Pipeline

1. **Train/Test Split**  
   - 80% train / 20% test  
2. **Classifier** (XGBoost)  
   - Grid search over `n_estimators` & `max_depth`  
   - Metrics: **Accuracy**, **ROC AUC**  
   - Saved as `Model/xgb_model.joblib`  
3. **Regressor** (XGBoost Regressor)  
   - Predicts `Profit Change` directly  
   - Metrics: **RMSE**, **MAE**, **MAPE**  
4. **Outputs** written to `outputs/`:  
   - `metrics.txt` (all five numbers)  
   - `line_actual_vs_pred_zoomed_fixed.png`  
   - `actual_vs_pred_zoomed_scatter.png`  
   - `residuals_filtered.png`  

---

## ⚡ Usage

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt


2. **Train & evaluate**

   ```bash
   python Model/Training.py
   ```

   → Check `outputs/` for metrics & charts.
3. **Predict on new data**

   * Prepare `Data/new_companies.csv` with headers:

     ```
     Company Name,Profit 2022,Profit 2024,Sector,Customer_Focused
     ```
   * Run:

     ```bash
     python Model/Prediction.py
     ```

   → `outputs/predictions.csv` will list each company’s 0/1 “WillBenefit.”

---

🤔 Future Work

* Add more features (market share, efficiency).
* Try other algorithms (LightGBM, CatBoost).
* Deploy as a small API (Flask or FastAPI).
* Build an auto-retraining pipeline for live data.

---


