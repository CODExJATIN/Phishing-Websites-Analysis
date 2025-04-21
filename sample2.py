# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from scipy.io import arff
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Load ARFF file
data, meta = arff.loadarff("Training_Dataset.arff")

# %%
df = pd.DataFrame(data)

# %%
# Decode byte columns if needed
df = df.map(lambda x: x.decode() if isinstance(x, bytes) else x)

# %%
# Preview
df.head()

# %%
print("Shape:", df.shape)
df.info()
df.describe()
df.isnull().sum()
df['Result'].value_counts()

# %%
# Visualization - Class Distribution
sns.countplot(x='Result', data=df)
plt.title("Phishing (-1) vs Legitimate (1)")
plt.show()

# %%
# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# %%
# Check correlations
correlation_matrix = df.corr()

# Remove highly correlated features (e.g., correlation > 0.9)
threshold = 0.9
correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

# Drop correlated features
df = df.drop(columns=correlated_features)

# %%
# Machine Learning Models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE

# %%
# #Drop unwanted features 
# drop_features = [
#     'having_At_Symbol',
#     'RightClick',
#     'SFH',
#     'Submitting_to_email',
#     'popUpWidnow',
#     'Iframe',
#     'Page_Rank'
# ]

# df = df.drop(columns=drop_features)

# %%
# Define features and target
X = df.drop(columns=['Result'])  # Assuming 'Result' is your target
y = df['Result']

# %%
# # Univariate feature selection
# k_best = SelectKBest(score_func=f_classif, k=25)
# X_kbest = k_best.fit_transform(X, y)

# # Get the selected feature names
# kbest_features = X.columns[k_best.get_support()].tolist()
# print("KBest selected features:", kbest_features)

# %%


# rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=20)
# rfe.fit(X, y)

# selected_features = X.columns[rfe.support_]
# print("Top selected features:\n", selected_features.tolist())

# %%
# # Reduce X to KBest features first
# X_reduced = X[kbest_features]

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# # Apply SMOTE to balance classes in training data only
from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

# # Random Forest for feature importance on top of KBest
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train, y_train)

# # Feature importance from RF
# importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)

# # Filter by importance
# threshold = 0.01
# final_features = importances[importances >= threshold].index.tolist()
# print("Final selected features (after KBest + RF):", final_features)

# # Final data
# X_train = X_train[final_features]
# X_test = X_test[final_features]
# Step 1: Fit each selection method and collect selected features

# SelectKBest
k_best = SelectKBest(score_func=f_classif, k=20)
k_best.fit(X, y)
kbest_features = set(X.columns[k_best.get_support()])

# RFE
rfe_model = RFE(estimator=RandomForestClassifier(), n_features_to_select=20)
rfe_model.fit(X, y)
rfe_features = set(X.columns[rfe_model.support_])

# RandomForest Feature Importance
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
rf_features = set(importances[importances >= 0.01].index)

# Combine using majority voting
all_features = list(X.columns)
final_features = []

for feature in all_features:
    votes = sum([
        feature in kbest_features,
        feature in rfe_features,
        feature in rf_features
    ])
    if votes >= 2:  # At least 2 out of 3 methods agree
        final_features.append(feature)

print("Final selected features (at least 2 methods agree):")
print(final_features)

# Filter X with final selected features
X_reduced = X[final_features]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# SMOTE to handle imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# %%
# Model training and evaluation
results = {}

# %%
# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
print("Confusion Matrix \n",confusion_matrix(y_test,log_pred))
print("Classification Rport \n",classification_report(y_test,log_pred))
results['Logistic Regression'] = classification_report(y_test, log_pred, output_dict=True)

# %%
# SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print("Confusion Matrix: \n",confusion_matrix(y_test,svm_pred))
print("Classification Report: \n",classification_report(y_test,svm_pred))
results['SVM'] = classification_report(y_test, svm_pred, output_dict=True)

# %%
# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("Confusion Matrix: \n" ,confusion_matrix(y_test,dt_pred))
print("Classification Report: \n",classification_report(y_test,dt_pred))
results['Decision Tree'] = classification_report(y_test, dt_pred, output_dict=True)

# %%
# KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
print("Confusion Matrix: \n" , confusion_matrix(y_test,knn_pred))
print("Classification Report: \n", classification_report(y_test,knn_pred))
results['KNN'] = classification_report(y_test, knn_pred, output_dict=True)

# %%
# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Confusion Matrix: \n" ,confusion_matrix(y_test,rf_pred))
print("Classification Report: \n",classification_report(y_test,rf_pred))
results['Random Forest'] = classification_report(y_test, rf_pred, output_dict=True)

# %%
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

# Naive Bayes Model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

# Confusion Matrix and Classification Report
print("Confusion Matrix: \n", confusion_matrix(y_test, nb_pred))
print("Classification Report: \n", classification_report(y_test, nb_pred))

# Store the results
results['Naive Bayes'] = classification_report(y_test, nb_pred, output_dict=True)

# %%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Gradient Boosting Model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# Confusion Matrix and Classification Report
print("Confusion Matrix: \n", confusion_matrix(y_test, gb_pred))
print("Classification Report: \n", classification_report(y_test, gb_pred))

# Store results
results['Gradient Boosting'] = classification_report(y_test, gb_pred, output_dict=True)

# %%
# Accuracy Summary
accuracy_scores = {model: round(metrics['accuracy'] * 100, 2) for model, metrics in results.items()}
accuracy_scores

# %%
# Print comparison in a nice format
for model, score in accuracy_scores.items():
    print(f"{model}: {score}% accuracy")

# %%
import matplotlib.pyplot as plt

# Assuming this is your dictionary
accuracy_scores = {model: round(metrics['accuracy'] * 100, 2) for model, metrics in results.items()}

# Plot
plt.figure(figsize=(10, 6))
plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color='skyblue', edgecolor='black')

# Labels & Title
plt.title("Model Accuracy Comparison", fontsize=16)
plt.xlabel("Models", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate bars with accuracy
for i, (model, acc) in enumerate(accuracy_scores.items()):
    plt.text(i, acc + 1, f"{acc}%", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# %%
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Create the base models
models = [
    ('logreg', LogisticRegression(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42)),
    ('svm', SVC(random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('knn', KNeighborsClassifier())
]

# Create a Voting Classifier (hard voting)
voting_model = VotingClassifier(estimators=models, voting='hard')

# Fit the voting classifier
voting_model.fit(X_train, y_train)

# Make predictions
voting_pred = voting_model.predict(X_test)

# Print classification report
print(classification_report(y_test, voting_pred))


# Print accuracy
accuracy = accuracy_score(y_test, voting_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# %%

# %%

# %%

# %%

# %%
