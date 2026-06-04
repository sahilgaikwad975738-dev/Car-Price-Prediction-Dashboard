import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# ── 1. Generate Data ──────────────────────────────────────────────────────────
np.random.seed(42)
N = 300

brand_mult = {'Maruti': 3.2, 'Hyundai': 4.1, 'Honda': 5.8, 'Toyota': 7.2}
fuel_mult  = {'Petrol': 1.0, 'Diesel': 1.12, 'Electric': 1.35}
owner_mult = {1: 1.0, 2: 0.82, 3: 0.68}

df = pd.DataFrame({
    'Brand':     np.random.choice(list(brand_mult), N),
    'Year':      np.random.randint(2012, 2025, N),
    'Fuel':      np.random.choice(list(fuel_mult), N),
    'KmDriven':  np.random.randint(5000, 180000, N),
    'Owners':    np.random.choice([1, 2, 3], N),
})

df['Price'] = df.apply(lambda r: round(
    brand_mult[r.Brand] * 1e5
    * np.exp(-0.11 * (2025 - r.Year))
    * np.exp(-0.0000025 * r.KmDriven)
    * fuel_mult[r.Fuel]
    * owner_mult[r.Owners]
    * np.random.normal(1, 0.07)
    / 1000) * 1000, axis=1).clip(lower=50000)

# ── 2. Encode & Split ─────────────────────────────────────────────────────────
for col in ['Brand', 'Fuel']:
    df[col] = LabelEncoder().fit_transform(df[col])

X, y = df.drop('Price', axis=1), df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 3. Train Model ────────────────────────────────────────────────────────────
model = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── 4. Metrics ────────────────────────────────────────────────────────────────
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score : {r2:.3f}")
print(f"MAE      : ₹{mae:,.0f}")
print(f"Records  : {N}")

# ── 5. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Feature Importance
pd.Series(model.feature_importances_, index=X.columns) \
    .sort_values() \
    .plot.barh(ax=axes[0], color='#378ADD', title='Feature Importance')

# Actual vs Predicted
axes[1].scatter(y_test / 1000, y_pred / 1000, alpha=0.4, s=14, color='#1D9E75')
axes[1].set(title='Actual vs Predicted', xlabel='Actual (₹K)', ylabel='Predicted (₹K)')

# Avg Price by Year
df.groupby('Year')['Price'].mean().div(1000) \
    .plot(ax=axes[2], marker='o', color='#BA7517', title='Avg Price by Year')
axes[2].set_ylabel('Price (₹K)')

plt.tight_layout()
plt.savefig('car_price_charts.png', dpi=150)
plt.show()
