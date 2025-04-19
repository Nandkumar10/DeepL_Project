import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np 
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder

## Load the trained model
#model = tensorflow.keras.models.load_mdodel('model.h5')

# Load the pkl fils
with open('label_encoder_gen.pkl','rb') as file:
        label_encoder_gen = pickle.load(file)

with open('onehote_encoder_geo.pkl','wb') as file:
        onehot_encoder_geo = pickle.load(file)

with open('scale.pkl','wb') as file:
        scaler = pickle.load(file)

## Streamlit app
st.title('Customer Churn Predction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gen.classes_)
age = st.slider('Age',18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit_Score')
estimated_salary = st.number_input('Estimated_Salary')
num_of_products = st.slider('Num_Of_Products',1 ,4)
tenure = st.slider('Tenure',0 ,10)
has_cr_card = st.selectbox('Has_Cr_Card',[0, 1])
is_active_member = st.selectbox('Is_Active_Member',[0, 1])

## Preapare the input data
input_data = pd.DataFrame({
        'CreditScore': [credit_score],
	'Gender': [label_encoder_gen.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],	
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],	
        'EstimatedSalary': [estimated_salary]
}
)

# Onhot Encode for 'Geography'
geo_encode = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encode_df = pd.DataFrame(geo_encode, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_df = pd.DataFrame([input_data])

##Combine columns
input_df = pd.concat([input_df.reset_index(drop=True), geo_encode_df],axis=1)

##Scale the input df
input_df_scaled = scaler.transform(input_df)

#Predict Churn
predction = model.predict(input_df_scaled)
prd_prob = predction[0][0]

if prd_prob > 0.5:
        st.write('The customer is likely to churn.')
else:
        st.write('The customer is not likely to churn. ')