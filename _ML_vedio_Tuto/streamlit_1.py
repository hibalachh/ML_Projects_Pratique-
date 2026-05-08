import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

##Streamlit is a powerful and easy to use Python library for developing web based applications. 
# It does not require detailed knowledge of web frameworks, or web design,
#  and it is great for deploying machine learning models to your customers, colleagues and the world.

st.title('Data Explorer the upload data')
st.text('This is a web app to explore this data.')

upload_file = st.file_uploader('Upload ur file here!')

if upload_file:
#upload data if we upload it 
    df = pd.read_csv(upload_file)
#do statistcs description
    st.write(df.describe())

#affiche 5first ligne of data
    st.header('Data Header')
    st.write(df.head())

#visualisation avec scatter for heath with outcome
    fig, ax= plt.subplots(1,1)
    ax.scatter(x=df['health'],y=df['income'])
    ax.set_xlabel('health')
    ax.set_ylabel('income')
    st.pyplot(fig)


