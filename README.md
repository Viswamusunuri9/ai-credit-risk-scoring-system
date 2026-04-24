# 💳 AI-Powered Credit Risk Scoring System

An end-to-end machine learning system to assess loan default risk and support financial decision-making using probability-based risk scoring.

---

## 🚀 Overview

This project builds a complete credit risk assessment pipeline using machine learning, transforming raw applicant data into actionable lending decisions.

Unlike traditional classification projects, this system integrates:

* Risk probability estimation
* Threshold-based decision logic
* Business-aware trade-offs (precision vs recall)
* Interactive user interface for real-time assessment

---

## 🎯 Problem Statement

Financial institutions must evaluate whether a loan applicant is likely to default.

Challenges include:

* Minimizing financial loss (false negatives)
* Avoiding rejection of good customers (false positives)
* Making consistent, data-driven decisions

---

## 🧠 Solution Approach

### 🔹 Data Processing

* Cleaned and preprocessed German credit dataset
* Handled missing values and categorical variables

### 🔹 Feature Engineering

* Created key feature:

  * **Credit per Month = Credit Amount / Duration**

  This captures repayment burden — a critical indicator of risk.

---

### 🔹 Model Development

* Evaluated multiple models:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * XGBoost

* Final Model: **XGBoost**

* Evaluation Metric: **ROC-AUC (~0.93)**

---

### 🔹 Threshold Optimization (Key Insight)

Default classification threshold (0.5) was adjusted to **0.4** to:

* Increase recall (detect more risky applicants)
* Reduce financial risk from false negatives

---

## 📊 Key Results

* ROC-AUC: **~0.93**
* Improved risk detection via threshold tuning
* Built a decision system, not just a classifier

---

## 🖥️ Interactive Dashboard

Users can:

* Input applicant details
* Get default probability
* View risk classification (High / Low)
* Receive actionable recommendation (Approve / Reject)

---

## 💼 Business Impact

* Supports data-driven loan approvals
* Reduces potential financial loss
* Improves consistency in decision-making
* Bridges ML predictions with business logic

---

## ⚙️ System Architecture

```id="cr-arch"
User Input → Feature Engineering → ML Model → Probability → Threshold Logic → Decision Output
```

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Streamlit

---

## 📂 Project Structure

```id="cr-struct"
credit-risk-system/
│
├── data/
├── images/
├── models/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│
├── app.py
├── requirements.txt
├── README.md
```

---

## ▶️ How to Run

### 1. Clone repo

```id="cr-run1"
git clone <your-repo-link>
cd credit-risk-system
```

### 2. Install dependencies

```id="cr-run2"
pip install -r requirements.txt
```

### 3. Train model

```id="cr-run3"
cd src
python train.py
```

### 4. Run app

```id="cr-run4"
streamlit run app.py
```

---

## 🔮 Future Improvements

* SHAP-based explainability (feature-level insights)
* Real-time API deployment
* Integration with banking systems
* Risk segmentation (Low / Medium / High)

---

## 👤 Author

**Viswa Musunuri**
AI/ML Engineer | GenAI Enthusiast
