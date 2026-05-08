#Create multiple pages web app with streamlit

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px 

def stat(dataframe):
    st.header('Data Statistics')
    st.write(dataframe.describe())

def data_header(dataframe):
    st.header('Data Header')
    st.write(dataframe.head())  # Fixed: changed df to dataframe

def plot(dataframe):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x=dataframe['health'], y=dataframe['income'])  # Fixed: changed df to dataframe
    ax.set_xlabel('health')
    ax.set_ylabel('income')
    st.pyplot(fig)

def interactive_plot(dataframe):
    x_axis_val = st.selectbox('Select X-Axis Value', options=dataframe.columns)  # Fixed: changed df to dataframe and 'option=' to 'options='
    y_axis_val = st.selectbox('Select Y-Axis Value', options=dataframe.columns)  # Fixed: changed df to dataframe and 'option=' to 'options='
    
    plot = px.scatter(dataframe, x=x_axis_val, y=y_axis_val)  # Fixed: removed space before =, changed variable name to avoid conflict
    st.plotly_chart(plot)

st.title('Data Explorer the upload data')
st.text('This is a web app to explore this data.')

st.sidebar.title('Navigation')  # Fixed: spelling
upload_file = st.sidebar.file_uploader('Upload your file here!')  # Fixed: spelling

options = st.sidebar.radio('Pages', options=['Home', 'Data Statistics', 'Data Header', 'Plot', 'Interactive Plot'])  # Fixed: spelling

if upload_file:
    #upload data if we upload it 
    df = pd.read_csv(upload_file)

if options == 'Data Statistics':  # Fixed: spelling
    stat(df)
elif options == 'Data Header':
    data_header(df)
elif options == 'Plot':
    plot(df)
elif options == 'Interactive Plot':
    interactive_plot(df)  # Fixed: was calling plot(df) instead of interactive_plot(df)