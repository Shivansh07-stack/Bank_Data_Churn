# Bank Customer Churn Prediction Project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# Loading Dataset
df = pd.read_csv("Bank_Customer_Churn_Prediction.csv")

print(df.head())
print(df.info())
print(df.describe())

# Droping Unnecessary Columns
df = df.drop(['customer_id'], axis=1)

# Encoding Categorical Variables
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])  # Female = 0, Male = 1
df = pd.get_dummies(df, columns=['country'], drop_first=True)

# EDA(Visualizations)
sns.countplot(x='churn', data=df)
plt.title("Churn Count")
plt.show()

sns.boxplot(x='churn', y='age', data=df)
plt.title("Churn vs Age")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

