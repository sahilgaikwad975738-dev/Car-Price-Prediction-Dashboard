import streamlit as st
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

st.title("🚗 Car Price Prediction")

np.random.seed(42)
N = 300
df = pd.DataFrame({
    'Brand': np.random.choice(['Maruti','Hyundai','Honda','Toyota'], N),
    'Year':  np.random.randint(2012, 2025, N),
    'Fuel':  np.random.choice(['Petrol','Diesel','Electric'], N),
    'KmDriven': np.random.randint(5000, 180000, N),
    'Owners': np.random.choice([1,2,3], N),
})
bm = {'Maruti':3.2,'Hyundai':4.1,'Honda':5.8,'Toyota':7.2}
fm = {'Petrol':1.0,'Diesel':1.12,'Electric':1.35}
df['Price'] = df.apply(lambda r: round(
    bm[r.Brand]*1e5 * np.exp(-0.11*(2025-r.Year))
    * np.exp(-0.0000025*r.KmDriven) * fm[r.Fuel]
    * {1:1.0,2:0.82,3:0.68}[r.Owners] * np.random.normal(1,.07)
/1000)*1000, axis=1).clip(50000)

for c in ['Brand','Fuel']:
    le = LabelEncoder(); df[c] = le.fit_transform(df[c])
X, y = df.drop('Price',axis=1), df['Price']
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=.2,random_state=42)
m = GradientBoostingRegressor(n_estimators=150,max_depth=4,random_state=42).fit(X_tr,y_tr)

r2  = r2_score(y_te, m.predict(X_te))
mae = mean_absolute_error(y_te, m.predict(X_te))

col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.3f}")
col2.metric("MAE", f"₹{mae:,.0f}")
col3.metric("Records", N)

fig, ax = plt.subplots(1,3,figsize=(13,4))
pd.Series(m.feature_importances_,index=X.columns).sort_values().plot.barh(ax=ax[0],color='#378ADD',title='Feature Importance')
ax[1].scatter(y_te/1000,m.predict(X_te)/1000,alpha=.4,s=14,color='#1D9E75')
ax[1].set(title='Actual vs Predicted',xlabel='Actual ₹K',ylabel='Predicted ₹K')
df.groupby('Year')['Price'].mean().div(1000).plot(ax=ax[2],marker='o',color='#BA7517',title='Price by Year')
plt.tight_layout()
st.pyplot(fig)   # ← this is the fix
