import streamlit as st

st.title("My First Streamlit App")
st.write("This is a Streamlit app running from the Desktop!")

if st.button("click me"):
    st.write("Button clicked!") 




number = st.slider("pick a number",1,100)
st.write(f"you selected:{number}")

if st.checkbox("Show Greeting"):
   st.write("Hello, Streamlit user")
   st.balloons()


choice = st.radio("pick any number",["red","blue","green"])
st.write(f"You selected: {choice}")

name = st.text_input("Enter your name:")
if name:
    st.write(f"My name is:{name}")

upload_file = st.file_uploader("upload a file",type=["csv","text"])
if upload_file:
    st.write("upload file content")
    st.write(upload_file.getvalue().decode("utf-8"))


import pandas as pd
import numpy as np
st.title("Bar chart") 
data = pd.DataFrame(np.random.randn(50,2),columns=["x","y"])
st.bar_chart(data)   