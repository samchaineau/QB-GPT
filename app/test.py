import streamlit as st

number = st.number_input("Insert a number", value=None, placeholder="Type a number...", key = "1")
number = st.number_input("Insert a number", value=None, placeholder="Type a number...", key = "2")
st.write('The current number is ', number)