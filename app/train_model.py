import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 1. Load Data ----------
df = pd.read_csv('heart_cleveland_upload.csv')

TARGET_COL = 'condition'
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# ---------- 2. Preprocessing ----------
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ---------- 3. Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------- 4. Initial Model to Get Feature Importances ----------
# دقت کن برای رفع هشدار، مدل رو با numpy array آموزش میدیم:
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
base_model.fit(X_train.values, y_train)

# رسم اهمیت ویژگی‌ها با نام ستون‌ها (از DataFrame استفاده می‌کنیم)
importances = base_model.feature_importances_
feat_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
feat_df = feat_df.sort_values('importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feat_df)
plt.title("Feature Importance (Initial RF)")
plt.tight_layout()
plt.savefig('model/feature_importance.png')
plt.close()

# ---------- 4.5 Feature selection ----------
selector = SelectFromModel(base_model, threshold=-np.inf, max_features=8, prefit=True)

# حتما transform هم روی numpy array باشه:
X_train_sel = selector.transform(X_train.values)
X_test_sel = selector.transform(X_test.values)

selected_features = X.columns[selector.get_support()].tolist()
print("Selected Features:", selected_features)

# ---------- 5. GridSearchCV for Hyperparameter Tuning ----------
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)

grid_search.fit(X_train_sel, y_train)
best_model = grid_search.best_estimator_

print("Best Params:", grid_search.best_params_)

# ---------- 6. Evaluation ----------
y_pred = best_model.predict(X_test_sel)
print(classification_report(y_test, y_pred))

# ---------- 7. Save Model + Selected Features + Scaler ----------
os.makedirs('model', exist_ok=True)
joblib.dump(best_model, 'model/model.pkl')
joblib.dump(selector, 'model/feature_selector.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(selected_features, 'model/selected_features.pkl')
print("Model and preprocessing saved.")
