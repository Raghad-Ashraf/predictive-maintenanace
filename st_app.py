import joblib
import streamlit as st
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier
import xgboost as xgb
  
with open('model.pkl', 'rb') as file:
    model = joblib.load("model.pkl")   
st.set_page_config("Predictive Maintenance site", layout = 'wide')   
    
def home_page():   
    #Display text
    st.title("Predictive Maintenance")
    st.markdown("This synthetic dataset is modeled after an existing milling machine and consists of 10 000 data points stored as rows with     14 features in columns")

    #load data
    st.markdown("Here is the dataset:")
    df = pd.read_csv("Predictive Maintenance.csv")
    st.dataframe(df.head())
    st.markdown(" 1) UID: unique identifier ranging from 1 to 10000 ")
    st.markdown("2) product ID: consisting of a letter L, M, or H for low (50% of all products), medium (30%) and high (20%) as product         quality variants and a variant-specific serial number")
    st.markdown("3) type: just the product type L, M or H from column 2")
    st.markdown("4) air temperature [K]: generated using a random walk process later normalized to a standard deviation of 2 K around 300   K")
    st.markdown("5) process temperature [K]: generated using a random walk process normalized to a standard deviation of 1 K, added to the    air temperature plus 10 K")
    st.markdown("6) rotational speed [rpm]: calculated from a power of 2860 W, overlaid with a normally distributed noise")
    st.markdown("7) torque [Nm]: torque values are normally distributed around 40 Nm with a SD = 10 Nm and no negative values")
    st.markdown("8) tool wear [min]: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process.")
    st.markdown("9) a 'machine failure' label that indicates, whether the machine has failed in this particular datapoint for any of the   following failure modes are true.")
def faliure():
    st.markdown("The machine failure consists of five independent failure modes:")
    st.markdown(" 1) tool wear failure (TWF): the tool will be replaced of fail at a randomly selected tool wear time between 200 - 240 mins (120 times in our dataset). At this point in time, the tool is replaced 69 times, and fails 51 times (randomly assigned).tool wear failure (TWF): the tool will be replaced of fail at a randomly selected tool wear time between 200 - 240 mins (120 times in our dataset). At this point in time, the tool is replaced 69 times, and fails 51 times (randomly assigned).")
    st.markdown("2) heat dissipation failure (HDF): heat dissipation causes a process failure, if the difference between air- and process temperature is below 8.6 K and the tools rotational speed is below 1380 rpm. This is the case for 115 data points")
    st.markdown(" 3) power failure (PWF): the product of torque and rotational speed (in rad/s) equals the power required for the process. If this power is below 3500 W or above 9000 W, the process fails, which is the case 95 times in our dataset.")
    st.markdown(" 4) overstrain failure (OSF): if the product of tool wear and torque exceeds 11,000 minNm for the L product variant (12,000 M, 13,000 H), the process fails due to overstrain. This is true for 98 datapoints.")
    st.markdown("5) random failures (RNF): each process has a chance of 0,1 % to fail regardless of its process parameters. This is the case for only 5 datapoints, less than could be expected for 10,000 datapoints in our dataset.")
    st.markdown("If at least one of the above failure modes is true, the process fails and the 'machine failure' label is set to 1. It is therefore not transparent to the machine learning method, which of the failure modes has caused the process to fail")
def inputs():
    st.title("Predict")
    
    Air_temperature = st.number_input("Air temperature [K]",  key="air_temp")
    Process_temperature = st.number_input("Process temperature [K]", key = "process_temp")
    Rotational_speed = st.number_input("Rotational speed [rpm]",  key="rotational speed")
    Torque = st.number_input("Torque [Nm] ", key = "Torque " )
    Tool_wear = st.number_input("Tool wear [min]", key="tool wear")
    
    if st.button("Predict"):
        predict(Air_temperature, Process_temperature, Rotational_speed, Torque, Tool_wear)

def predict(Air_temperature, Process_temperature, Rotational_speed, Torque, Tool_wear):
    # Use the inputs directly in the predict function
    input_data = [[Air_temperature, Process_temperature, Rotational_speed, Torque, Tool_wear]]
    pred = model.predict(input_data)
    st.write(f'The prediction is: {pred}')
    st.markdown("if 1 failure")
    
page = st.sidebar.selectbox("select page", ["Home", "machine faliure modes", "predict"])
if page == 'Home':
    home_page()
elif page == 'machine faliure modes':
    faliure()
else:
    inputs()
    
    
