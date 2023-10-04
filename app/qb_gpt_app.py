import streamlit as st
import plotly.graph_objs as go

from app_tools import set_app_title_and_logo, qb_gpt_page, helenos_page, about_page, contacts_and_disclaimers


# Define the main function to run the app
def main():
    set_app_title_and_logo()
    
    # Create a sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ("About", "QB-GPT", "Helenos", "Contacts and Disclaimers"))

    if page == "About":
        # Page 1: Contacts and Explanations
        about_page()
       

    elif page == "QB-GPT":
        # Page 2: QB-GPT
        st.title("QB-GPT")
        qb_gpt_page()
        # Add input buttons and fields specific to QB-GPT page
        # Display a simple Plotly visualization

    elif page == "Helenos":
        # Page 3: Helenos
        st.title("Helenos")
        helenos_page()
        # Add input buttons and fields specific to Helenos page
        # Display a Plotly histogram visualization
        
    if page == "Contacts and Disclaimers":
        # Page 1: Contacts and Explanations
        contacts_and_disclaimers()
        
        
if __name__ == "__main__":
    main()