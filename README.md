# 📊 Decision Tree on Imbalanced Data

This project builds and evaluates a **Decision Tree classifier** to predict company bankruptcy using financial data from Poland. The dataset is highly imbalanced, so both **undersampling** and **oversampling** techniques were tested to improve performance.

---

## 📁 Project Structure

---

## 📌 Dataset

- **Target Variable**: `bankrupt` (1 = bankrupt, 0 = not bankrupt)
- **Features**: Financial ratios and performance metrics (e.g., `feat_27` = profit/expenses ratio)

---

## ⚖️ Handling Class Imbalance

The dataset was heavily imbalanced:
- **Baseline Accuracy** (predicting only the majority class): `95.19%`

To address this, we applied:
- **Undersampling** (via `RandomUnderSampler`)
- **Oversampling** (assumed to be via `RandomOverSampler` or SMOTE)

---

## 🧪 Models & Evaluation

Three Decision Tree models were trained:

| Model         | Training Accuracy | Test Accuracy |
|---------------|-------------------|---------------|
| `model_reg` (original imbalanced) | 1.0000            | 0.9359        |
| `model_under` (undersampled)      | 0.7421            | 0.7104        |
| `model_over` (oversampled)        | 1.0000            | 0.9344        |

### 🔍 Insights:
- The **baseline model** (no sampling) overfits but performs reasonably on the test set.
- **Undersampling** helps reduce overfitting but lowers accuracy due to less training data.
- **Oversampling** retains more data and generalizes well (similar to the original).

---

## 📈 Visualizations

- Class balance bar plot
- Feature distributions (histograms, boxplots)
- Correlation heatmap
- Confusion matrix

---

## 🛠️ Tools & Libraries

- Python, Pandas, scikit-learn, imbalanced-learn
- Seaborn, Matplotlib
- DecisionTreeClassifier
- RandomUnderSampler / RandomOverSampler

---


python src/predict.py
