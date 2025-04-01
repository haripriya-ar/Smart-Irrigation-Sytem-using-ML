import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px

file="data.csv"
df=pd.read_csv(file)
st.title("Irrigation Pump Prediction Model with Random Forest Classifier")
st.subheader("Dataset Overview")
# Encode the categorical 'crop' column using one-hot encoding if it exists
if 'crop' in df.columns:
    df_encoded=pd.get_dummies(df,columns=['crop'],drop_first=True)
else:
    df_encoded=df
# Define features (X) and target (y)
X = df_encoded.drop(columns=['pump'],errors='ignore')  # Avoid error if 'pump' doesn't exist
y = df_encoded.get('pump', pd.Series())  # Use empty Series if 'pump' doesn't exist

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

joblib.dump(model,'irrigation_model_with_crop.pkl')

y_pred=model.predict(X_test)

# Model Evaluation
accuracy=accuracy_score(y_test,y_pred)
st.subheader(f"Model Accuracy: {accuracy:.4f}")
cm=confusion_matrix(y_test,y_pred)
st.subheader("Confusion Matrix")
fig,ax=plt.subplots(figsize=(6, 6))
sns.heatmap(cm, annot=True,fmt="d",cmap="Blues",xticklabels=['OFF','ON'],yticklabels=['OFF','ON'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
st.pyplot(fig)
# Classification Report
st.subheader("Classification Report")
st.text(classification_report(y_test,y_pred,target_names=['OFF','ON']))

st.subheader("Feature Importance")
feature_importance = model.feature_importances_
fig, ax = plt.subplots(figsize=(8,6))
plt.bar(X.columns,feature_importance)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
st.pyplot(fig)

st.subheader("Make a Prediction")
moisture = st.slider("Moisture Level",min_value=0,max_value=1000)
temperature = st.slider("Temperature",min_value=0,max_value=50)
crop_type = st.selectbox("Select Crop",options=['Cotton','Wheat'])
# Encode the selected crop type
if 'crop' in df.columns:
    crop_encoded=pd.get_dummies(pd.DataFrame([crop_type],columns=['crop']),drop_first=True)
    new_data=np.array([[moisture,temperature]+crop_encoded.iloc[0].tolist()])
else:
    new_data=np.array([[moisture,temperature]])
if st.button("Predict Pump Status"):
    prediction=model.predict(new_data)
    prediction_label='ON' if prediction==1 else 'OFF'
    st.write(f"Predicted Pump Action: {prediction_label}")

st.subheader("Upload Your Own Dataset")
uploaded_file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith("csv"):
        uploaded_data = pd.read_csv(uploaded_file)
    else:
        uploaded_data = pd.read_excel(uploaded_file)
    st.write(uploaded_data.head())
    st.subheader("Make Predictions on Your Data")
    # Encode the uploaded data if crop column exists
    if 'crop' in uploaded_data.columns:
        uploaded_data_encoded = pd.get_dummies(uploaded_data, columns=['crop'], drop_first=True)
    else:
        uploaded_data_encoded = uploaded_data
    X_upload = uploaded_data_encoded.drop(columns=['pump'], errors='ignore')
    predictions = model.predict(X_upload)
    uploaded_data['Prediction'] = ['ON' if p == 1 else 'OFF' for p in predictions]
    st.write(uploaded_data[['pump', 'Prediction']])
    st.subheader("Download Prediction Results")
    st.download_button(
        label="Download Predicted Data",
        data=uploaded_data.to_csv(index=False),
        file_name="predicted_data.csv",
        mime="text/csv"
    )
# Hyperparameter tuning: Allow users to change the number of trees in the forest
st.subheader("Hyperparameter Tuning")
n_estimators = st.slider("Number of Trees in Random Forest", min_value=10, max_value=200, step=10, value=100)
if st.button("Retrain Model"):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    st.success("Model retrained successfully!")
    st.write(f"New Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
st.subheader("Data Visualization")
# Plot moisture vs temperature (scatter plot)
if 'moisture' in df.columns and 'temperature' in df.columns:
    fig = px.scatter(df,x="moisture",y="temperature",title="Moisture vs Temperature")
    st.plotly_chart(fig)
else:
    st.error("The dataset is missing the columns: 'moisture' and 'temperature'.")
st.subheader("Pump Status Distribution")
fig2=px.pie(df,names='pump',title="Distribution of Pump Status")
st.plotly_chart(fig2)
