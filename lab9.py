import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

file_path = 'titanic_prepared.csv'
df = pd.read_csv(file_path)

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"Decision Tree Accuracy: {dt_accuracy}")

xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Accuracy: {xgb_accuracy}")

lr_model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_accuracy}")

correlations = df.corr()['label'].abs().sort_values(ascending=False)
selected_features_alt = correlations[1:3].index.tolist()
dt_model_selected_alt = DecisionTreeClassifier(random_state=42)
dt_model_selected_alt.fit(X_train[selected_features_alt], y_train)
dt_pred_selected_alt = dt_model_selected_alt.predict(X_test[selected_features_alt])
dt_accuracy_selected_alt = accuracy_score(y_test, dt_pred_selected_alt)
print(f"Decision Tree Accuracy (Two Features): {dt_accuracy_selected_alt}")