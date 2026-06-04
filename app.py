import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("Car Price Prediction Dashboard")

# ── Generate Data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    np.random.seed(42)
    N = 300
    brand_mult = {'Maruti': 3.2, 'Hyundai': 4.1, 'Honda': 5.8, 'Toyota': 7.2}
    fuel_mult  = {'Petrol': 1.0, 'Diesel': 1.12, 'Electric': 1.35}
    owner_mult = {1: 1.0, 2: 0.82, 3: 0.68}

    df = pd.DataFrame({
        'Brand':    np.random.choice(list(brand_mult), N),
        'Year':     np.random.randint(2012, 2025, N),
        'Fuel':     np.random.choice(list(fuel_mult), N),
        'KmDriven': np.random.randint(5000, 180000, N),
        'Owners':   np.random.choice([1, 2, 3], N),
    })

    df['Price'] = df.apply(lambda r: round(
        brand_mult[r.Brand] * 1e5
        * np.exp(-0.11 * (2025 - r.Year))
        * np.exp(-0.0000025 * r.KmDriven)
        * fuel_mult[r.Fuel]
        * owner_mult[r.Owners]
        * np.random.normal(1, 0.07)
        / 1000) * 1000, axis=1).clip(lower=50000)
    return df

@st.cache_data
def train_model(df):
    df2 = df.copy()
    for col in ['Brand', 'Fuel']:
        df2[col] = LabelEncoder().fit_transform(df2[col])
    X, y = df2.drop('Price', axis=1), df2['Price']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return model, X, X_te, y_te, y_pred

df = load_data()
model, X, X_te, y_te, y_pred = train_model(df)

r2  = r2_score(y_te, y_pred)
mae = mean_absolute_error(y_te, y_pred)

# ── Metrics ────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.3f}")
col2.metric("MAE", f"₹{mae:,.0f}")
col3.metric("Records", len(df))

st.markdown("---")

# ── Charts ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Feature Importance
pd.Series(model.feature_importances_, index=X.columns) \
    .sort_values() \
    .plot.barh(ax=axes[0], color='#378ADD')
axes[0].set_title('Feature Importance')

# Actual vs Predicted
axes[1].scatter(y_te / 1000, y_pred / 1000, alpha=0.4, s=14, color='#1D9E75')
axes[1].set_title('Actual vs Predicted')
axes[1].set_xlabel('Actual (₹K)')
axes[1].set_ylabel('Predicted (₹K)')

# Avg Price by Year
df.groupby('Year')['Price'].mean().div(1000) \
    .plot(ax=axes[2], marker='o', color='#BA7517')
axes[2].set_title('Avg Price by Year')
axes[2].set_ylabel('Price (₹K)')

plt.tight_layout()
st.pyplot(fig)
