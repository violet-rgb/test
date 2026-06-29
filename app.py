import pickle
import pandas as pd
import streamlit as st

MODEL_PATH = 'best_model.pkl'

@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

model_bundle = load_model(MODEL_PATH)
model = model_bundle['model']
features = model_bundle['features']
scaler = model_bundle['scaler']

st.set_page_config(page_title='Heart Disease Predictor', page_icon='❤️')
st.title('Heart Disease Prediction')
st.write('Enter patient data to predict the likelihood of heart disease.')

inputs = {}
for feature in features:
    inputs[feature] = st.number_input(feature, value=0.0, format='%.4f')

if st.button('Predict'):
    input_df = pd.DataFrame([inputs])
    try:
        scaled_input = scaler.transform(input_df)
    except Exception as exc:
        st.error(f'Error applying scaler: {exc}')
    else:
        prediction = model.predict(scaled_input)
        proba = model.predict_proba(scaled_input)[:, 1] if hasattr(model, 'predict_proba') else None
        st.write('## Prediction Result')
        st.write('**Predicted class:**', int(prediction[0]))
        if proba is not None:
            st.write('**Predicted probability of positive class:**', float(proba[0]))
        st.success('Prediction completed successfully.')
