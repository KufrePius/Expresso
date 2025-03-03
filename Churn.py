import streamlit as st
import pandas as pd
#import matplotlib as plt
import joblib
import warnings
warnings.filterwarnings('ignore')
#import plotly as px

data = pd.read_csv('expresso_cleaned.csv')

st.markdown("<h1 style = 'color:rgb(74, 204, 204); text-align: center; font-size: 60px; font-family: Monospace'>CHURN PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color:rgb(196, 152, 193); text-align: center; font-family: Serif '>Built by KufreKing</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)

st.image('pngwing.com (3).png', caption= 'Built by KufreKing')


st.markdown("<h2 style = 'color: #F7C566; text-align: center; font-family: montserrat '>Purpose Of Study</h2>", unsafe_allow_html = True)
st.markdown("The purpose of this study is to develop a churn predictor app that identifies customers at risk of leaving. By analyzing historical data, the app will detect patterns and key factors influencing churn. Using a machine learning model, it will predict churn likelihood and provide insights for targeted retention strategies. This helps optimize customer retention, reduce acquisition costs, and support data-driven business decisions through an intuitive interface.")

#sidebar design
st.sidebar.image('user icon (2).png')


st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.divider() #seperates the background of study from the project data
st.header('Project Data')
st.dataframe(data, use_container_width= True)

#user input section
data_vol = st.sidebar.number_input('DATA_VOLUME', data['DATA_VOLUME'].min(), data['DATA_VOLUME'].max())
on_net = st.sidebar.number_input('ON_NET', data['ON_NET'].min(), data['ON_NET'].max())
reg= st.sidebar.number_input('REGULARITY', data['REGULARITY'].min(), data['REGULARITY'].max())
rev = st.sidebar.number_input('REVENUE', data['REVENUE'].min(), data['REVENUE'].max())
freq = st.sidebar.number_input('FREQUENCE', data['FREQUENCE'].min(), data['FREQUENCE'].max())
freq_rech = st.sidebar.number_input('FREQUENCE_RECH', data['FREQUENCE_RECH'].min(), data['FREQUENCE_RECH'].max())
mont = st.sidebar.number_input('MONTANT', data['MONTANT'].min(), data['MONTANT'].max())

selectedcol = ['DATA_VOLUME', 'CHURN',	'ON_NET', 'REGULARITY', 'REVENUE','FREQUENCE', 'MONTANT', 'FREQUENCE_RECH']
#user input linked to be the same with what is on the dataFrame
#users input
input_var = pd.DataFrame()
input_var['DATA_VOLUME'] = [data_vol]
input_var['ON_NET'] = [on_net]
input_var['REGULARITY'] = [reg]
input_var['REVENUE'] = [rev]
input_var['FREQUENCE'] = [freq]
input_var['MONTANT'] = [mont]
input_var['FREQUENCE_RECH'] = [freq_rech]


#Table to display the results of the users input
st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('User Inputs')
st.dataframe(input_var, use_container_width= True)

#Load selected encoded and scaled columns
data_vol = joblib.load('DATA_VOLUME_scaler.pkl')
mont = joblib.load('MONTANT_scaler.pkl')
rev= joblib.load('REVENUE_scaler.pkl')



#transform the users input with the imported scalers
input_var['DATA_VOLUME']= data_vol.transform(input_var[['DATA_VOLUME']])
input_var['MONTANT']= mont.transform(input_var[['MONTANT']])
input_var['REVENUE']= rev.transform(input_var[['REVENUE']])


#Bringing the model for prediction
model = joblib.load('Churn_predictormodel.pk1')
predict = model.predict(input_var)

if st.button('Check Your Customer Retention Status'):
    if predict[0] == 1:
        st.error(f'Unfortunately.... Your likely going to loose this customer, implement the customer retention failproof strategy')
        st.image('lost.png', width= 300)
    else:
        st.success(f'Congratulations.... This customer is secured, add him/her to the customer royalty program ')
        st.image('retainer.png', width= 300)
        st.balloons()