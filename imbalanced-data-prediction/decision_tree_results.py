# Imbalanced Data
# ==============================
# 📦 Import Libraries
# ==============================
# Core libraries for data handling, plotting, modeling, and resampling
import gzip
import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier


# ==============================
# 📥 Load and Clean Data
# ==============================
def wrangle(filename):
    # Load compressed JSON data and convert it into a DataFrame
    with gzip.open(filename, "r") as f:
        data = json.load(f)

    # Convert to DataFrame and set 'company_id' as the index
    df = pd.DataFrame().from_dict(data["data"]).set_index("company_id")
    return df

# Load the dataset using the wrangle function
df = wrangle("data/poland-bankruptcy-data-2009.json.gz")
print(df.shape)
df.head()


# ==============================
# 📊 Basic Data Exploration
# ==============================

# Show structure and types of data
df.info()

# Plot class balance (bankrupt vs. not bankrupt)
df["bankrupt"].value_counts(normalize=True).plot(
    kind="bar",
    xlabel="Bankrupt",
    ylabel="Frequency",
    title="Class Balance"
)


# ==============================
# 📈 Feature Distribution Analysis
# ==============================

# Summary statistics for feature 'feat_27'
df["feat_27"].describe().apply("{0:,.0f}".format)

# Plot histogram for 'feat_27'
df["feat_27"].hist()
plt.xlabel("POA / financial expenses")
plt.ylabel("Count")
plt.title("Distribution of Profit/Expenses Ratio")


# ==============================
# 📦 Outlier Clipping (10th–90th Percentile)
# ==============================

# Create a mask to filter out outliers in 'feat_27'
q1, q9 = df["feat_27"].quantile([0.1, 0.9])
mask = df["feat_27"].between(q1, q9)
mask.head()


# ==============================
# 📦 Boxplot: Feature vs Target
# ==============================

# Boxplot of 'feat_27' grouped by bankruptcy status (using clipped data)
sns.boxplot(x="bankrupt", y="feat_27", data=df[mask])
plt.xlabel("Bankrupt")
plt.ylabel("POA / financial expenses")
plt.title("Distribution of Profit/Expenses Ratio, by Bankruptcy Status")


# ==============================
# 📊 Correlation Matrix (All Features)
# ==============================

# Drop target column, calculate correlations between features
corr = df.drop(columns="bankrupt").corr()
sns.heatmap(corr)


# ==============================
# 🧾 Define Features and Target
# ==============================

target = "bankrupt"
X = df.drop(columns=target)
y = df[target]

print("X shape:", X.shape)
print("y shape:", y.shape)


# ==============================
# ✂️ Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ==============================
# 🔽 Apply Undersampling to Balance Classes
# ==============================

under_sampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = under_sampler.fit_resample(X_train, y_train)
print(X_train_under.shape)
X_train_under.head()

# Show class distribution after undersampling
y_train_under.value_counts(normalize=True)

# ==============================
# 🔽 Apply Oversampling to Balance Classes
# ==============================
over_sampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
print(X_train_over.shape)
X_train_over.head()

# Show class distribution after Oversampling
y_train_under.value_counts(normalize=True)

# ==============================
# 📏 Baseline Accuracy (Majority Class)
# ==============================

acc_baseline = y_train.value_counts(normalize=True).max()
print("Baseline Accuracy:", round(acc_baseline, 4))


# ==============================
# 🌳 Train Models with Different Sampling Strategies
# ==============================

# Train model on original (imbalanced) data
model_reg = make_pipeline(
    SimpleImputer(strategy="median"),
    DecisionTreeClassifier(random_state=42)
)
model_reg.fit(X_train, y_train)

# Train model on undersampled data
model_under = make_pipeline(
    SimpleImputer(strategy="median"),
    DecisionTreeClassifier(random_state=42)
)
model_under.fit(X_train_under, y_train_under)

# (Assumes X_train_over, y_train_over already defined elsewhere — SMOTE or oversampling should be done above)
model_over = make_pipeline(
    SimpleImputer(strategy="median"),
    DecisionTreeClassifier(random_state=42)
)
model_over.fit(X_train_over, y_train_over)

# ==============================
# ✅ Evaluate All Models (Accuracy)
# ==============================

for m in [model_reg, model_under, model_over]:
    acc_train = m.score(X_train, y_train)
    acc_test = m.score(X_test, y_test)

    print("Training Accuracy:", round(acc_train, 4))
    print("Test Accuracy:", round(acc_test, 4))

# ==============================
# 📊 Confusion Matrix for Original Model
# ==============================

ConfusionMatrixDisplay.from_estimator(model_reg, X_test, y_test)


# ==============================
# 🧠 View Tree Depth of Oversampled Model
# ==============================

depth = model_over.named_steps["decisiontreeclassifier"].get_depth()
print(depth)


