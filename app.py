import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Flower Classifier", page_icon="🌸")
st.title("🌸 Synthetic Flower Species Classifier")
st.write("SVM with tuned RBF kernel")

model = joblib.load("svm_flower_model.joblib")


def sync_input(source_key, target_key):
    st.session_state[target_key] = st.session_state[source_key]


def feature_input(label, minimum, maximum, default):
    slider_key = f"{label}_slider"
    number_key = f"{label}_number"
    st.session_state.setdefault(slider_key, default)
    st.session_state.setdefault(number_key, default)
    st.slider(
        label,
        minimum,
        maximum,
        step=0.01,
        key=slider_key,
        on_change=sync_input,
        args=(slider_key, number_key),
    )
    st.number_input(
        f"{label} (กรอกตัวเลข)",
        minimum,
        maximum,
        step=0.01,
        format="%.2f",
        key=number_key,
        on_change=sync_input,
        args=(number_key, slider_key),
    )
    return st.session_state[number_key]


sepal_length = feature_input("Sepal length (cm)", 4.0, 8.0, 5.8)
sepal_width = feature_input("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = feature_input("Petal length (cm)", 1.0, 7.0, 4.0)
petal_width = feature_input("Petal width (cm)", 0.1, 2.8, 1.3)

input_df = pd.DataFrame([{
"sepal_length_cm": sepal_length,
"sepal_width_cm": sepal_width,
"petal_length_cm": petal_length,
"petal_width_cm": petal_width
}])

if st.button("Predict species"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted species: {prediction}")