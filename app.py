import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Flower Classifier", page_icon="🌸")
st.title("🌸 Synthetic Flower Species Classifier")
st.write("SVM with tuned RBF kernel")

model = joblib.load("svm_flower_model.joblib")
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0, 0.1)
petal_width = st.slider("Petal width (cm)", 0.1, 2.8, 1.3, 0.1)

input_df = pd.DataFrame([{
"sepal_length_cm": sepal_length,
"sepal_width_cm": sepal_width,
"petal_length_cm": petal_length,
"petal_width_cm": petal_width
}])

if st.button("Predict species"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted species: {prediction}")