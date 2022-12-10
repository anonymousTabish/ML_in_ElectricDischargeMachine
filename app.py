import streamlit as st
import pickle
import numpy as np
import sklearn

# import the model
reg_rf = pickle.load(open('reg_rf.pkl','rb'))
reg_rf2 = pickle.load(open('reg_rf2.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))



html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">ML App for EDM </h2>
<h3 style="color:white;text-align:center;">(EN31 STEEL)</h3>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)
st.title("Material Removal Rate(MRR) & Surface Roughness(Ra) Predictor : ")


# Pulse_on_time
Pulse_on_time = st.number_input('Pulse_on_time')

#Pulse_off_time
Pulse_off_time = st.number_input('Pulse_off_time')

#Discharge_current
Discharge_current = st.number_input('Discharge_current')

# Voltage
Voltage = st.selectbox('Voltage(V)',[10,20,30,40,50,60,70,80,90,100,110,120])


if st.button('Predict MRR'):
    query = np.array([Pulse_on_time,Pulse_off_time,Discharge_current,Voltage])
    final_query = scalar.transform(query.reshape(1,4))
    st.title("Predicted Value of MRR is " + str(reg_rf.predict(final_query)[0]) + " gm/min")

if st.button('Predict Ra'):
    query = np.array([Pulse_on_time,Pulse_off_time,Discharge_current,Voltage])
    fin_query = query.reshape(1,4)
    st.title("Predicted Value of Ra is " + str(reg_rf2.predict(fin_query)[0]) + " um")