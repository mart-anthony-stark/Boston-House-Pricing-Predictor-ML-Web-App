import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Boston House Price Prediction App
This app predicts the **Boston House Price**!
""")
st.write('---')

# Load boston house dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns=["MEDV"])

st.sidebar.header('Input Parameters')

def user_input_features():
  CRIM = st.sidebar.slider('CRIM', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
  ZN = st.sidebar.slider('ZN', X.ZN.min(), X.ZN.max(), X.ZN.mean())
  INDUS = st.sidebar.slider('INDUS', X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
  CHAS = st.sidebar.slider('CHAS', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
  NOX = st.sidebar.slider('NOX', X.NOX.min(), X.NOX.max(), X.NOX.mean())
  RM = st.sidebar.slider('RM', X.RM.min(), X.RM.max(), X.RM.mean())
  AGE = st.sidebar.slider('AGE', X.AGE.min(), X.AGE.max(), X.AGE.mean())
  DIS = st.sidebar.slider('DIS', X.DIS.min(), X.DIS.max(), X.DIS.mean())
  RAD = st.sidebar.slider('RAD', X.RAD.min(), X.RAD.max(), X.RAD.mean())
  TAX = st.sidebar.slider('TAX', X.TAX.min(), X.TAX.max(), X.TAX.mean())
  PTRATIO = st.sidebar.slider('PTRATIO', X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
  B = st.sidebar.slider('B', X.B.min(), X.B.max(), X.B.mean())
  LSTAT = st.sidebar.slider('LSTAT', X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())
  data = {'CRIM': CRIM,
          'ZN': ZN,
          'INDUS': INDUS,
          'CHAS': CHAS,
          'NOX': NOX,
          'RM': RM,
          'AGE': AGE,
          'DIS': DIS,
          'RAD': RAD,
          'TAX': TAX,
          'PTRATIO': PTRATIO,
          'B': B,
          'LSTAT': LSTAT}
  features = pd.DataFrame(data, index=[0])
  return features

df = user_input_features()
# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build model
model = RandomForestRegressor()
model.fit(X, y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')