import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image
from tools import generator

def set_app_title_and_logo():
    st.set_page_config(
        page_title="QB-GPT",
        page_icon=":rocket:",
        layout="wide",
    )

def qb_gpt_page(ref_df, ref, tokenizer, model):
    
    with st.container():
        cola, colb = st.columns(2)
        with cola:
            selected_gameId = st.selectbox("Select Game ID", ref_df['gameId'].unique())
            filtered_df1 = ref_df[(ref_df['gameId'] == selected_gameId)]
        with colb:
            selected_Play= st.selectbox("Select Play ", filtered_df1['playId'].unique())
        filtered_df = filtered_df1[(filtered_df1['playId'] == selected_Play)].reset_index(drop ="True")

        # Display the filtered DataFrame
        st.write("Filtered Data:")
        st.dataframe(filtered_df)
    
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", 1.0, 10.0, 1.5, step = 0.5)
        with col2:
            n_select = st.slider("N movements to shortlist", 2, 100, 10, step = 1)
        
        QB_gen = generator(model=model,
                        tokenizer=tokenizer,
                        temp = temperature,
                        n_select = n_select)
    
    
    selected = filtered_df["index"][0]
    selection = ref[selected]
    
    colc, cold = st.columns(2)
    
    with colc:
        starts = st.slider("Temperature", 1, 21, 1, step = 1)
    with cold:
        frames = st.slider("n select", 1, 50, 20, step = 1)
    
    if st.button("Generate"):
        trial_d = QB_gen.tokenizer.truncate_to_time_t(selection, starts)
        generated = QB_gen.generate_sequence(trial_d, frames)
        decoded = QB_gen.tokenizer.decode_sequence(generated)

        step1 = QB_gen.prepare_for_plot(decoded)
        plot = pd.DataFrame(step1)

        decoded_true = QB_gen.tokenizer.decode_sequence(selection)
        step1_true = QB_gen.prepare_for_plot(decoded_true)
        plot_true = pd.DataFrame(step1_true)
        
        fig_gen = px.line(plot, x="input_ids_x", y="input_ids_y", animation_frame="pos_ids", color="OffDef", symbol="ids",
                        text="position_ids", title="Player Trajectories Over Time", line_shape="linear",
                        range_x=[0, 140], range_y=[0, 60], # Set X and Y axis ranges
                        render_mode="svg")  # Render mode for smoother lines

        # Customize the appearance of the plot
        fig_gen.update_traces(marker=dict(size=10), selector=dict(mode='lines'))
        fig_gen.update_layout(width=800, height=600) 
        st.plotly_chart(fig_gen)
        
        fig_true = px.line(plot_true, x="input_ids_x", y="input_ids_y", animation_frame="pos_ids", color="OffDef", symbol="ids",
                    text="position_ids", title="Player Trajectories Over Time",
                    range_x=[0, 140], range_y=[0, 60], # Set X and Y axis ranges
                    line_shape="linear",  # Draw lines connecting points
                    render_mode="svg")  # Render mode for smoother lines

        # Customize the appearance of the plot
        fig_true.update_traces(marker=dict(size=10), selector=dict(mode='lines'))
        fig_true.update_layout(width=800, height=600) 
        st.plotly_chart(fig_true)
    
        
def contacts_and_disclaimers():
    
    
    st.title("QB-GPT - Your Football Playbook Powerhouse!")
    
    qb_gpt_text_intro = """
    Are you a data scientist, a machine learning enthusiast, or simply a die-hard NFL fan looking to explore the power of Transformers in the world of American football? Look no further!
    """
    st.markdown(qb_gpt_text_intro)
    
    with st.expander("***What is QB-GPT?***"):
        
        qb_gpt_what = """
        QuarterBack-GPT (QB-GPT) is a companion in the world of football strategy and analytics. It's an innovative application with a model relying on the remarkable capabilities of Transformers to generate football plays that are not only strategic but also incredibly realistic. Imagine having an AI-powered coach in your corner, designing plays that could turn the tide of any game.
        """
        st.markdown(qb_gpt_what)
    
    with st.expander("***What's inside QB-GPT***"):
        
        qb_gpt_transf = """
        At the heart of QB-GPT lies the cutting-edge Transformer model, a deep learning architecture known for its prowess in understanding sequential data. It doesn't just create plays; it understands the game at a granular level, taking into account player positions, game situations, and historical data. It relies on the same conceptual approach behind the now famous "GPT" model of OpenAI. It's the playbook of the future, driven by the technology of tomorrow.
        
        A more detailed blogpost about the model QB-GPT can be found [here](link)
        """
        st.markdown(qb_gpt_transf)
        
    with st.expander("***QB-GPT in Action***"):
        
        qb_gpt_act = """
        With QB-GPT, you can explore a wide range of football scenarios. Design plays that are tailored to your team's strengths, simulate game situations, and experiment with different strategies—all at your fingertips. Whether you're a coach looking to refine your playbook or an NFL enthusiast seeking the thrill of strategic gameplay, QB-GPT has something for everyone.
        """
        st.markdown(qb_gpt_act)
        
        
    
    with st.expander("***Author***"):
        col1, col2 = st.columns([4, 1])
        with col1:
            author_text = """
            My name is Samuel Chaineau, I am 26 and live in Paris. This is my second work related to Deep-Learning applied to NFL NGS data. This work is released as an app in order to facilitate the interaction and feedbacks from user. I also hope having constructive academic/scientific feedbacks to improve the model and bring it to the next level. 
        
            I have a background in economics, management, statistics and computer sciences. I am currently the CTO of a french healthcare start-up called Nuvocare. Prior to that I worked 2 and a half years as a Data Scientist at Ekimetrics. 
        
            ***Contacts***
            
            If interested by the project, the app or wishing to discuss any related topics, feel free to contact me on :
            - My email : s.chaineau@roof.ai
            
            - Linkedin : [My profile](https://www.linkedin.com/in/samuel-chaineau-734b13122/)
            
            - X (Twitter, nobody says X) : [My profile](https://twitter.com/samboucon)
            
            - Or you can follow me on Medium : [My blog](https://medium.com/@sam.chaineau)
            """
            st.markdown(author_text)
        
        with col2:
            image = Image.open('app/assets/photo_cv.jpg')
            st.image(image)
        
    with st.expander("***Disclaimers***"):
        disclaimer_text = """
        This work is at a very early stage and while I think it shows promising results, I acknowledge that the model may yield disturbing results and potentially wrong.
        Maintaining and improving QB-GPT will be a long run. 

        I used data only found publicly on the internet (GitHub and Kaggle). I don't hold any relationship with NFL officials or any NFL teams. 
        I do not intend to have any payments or commercial activities via this app. It is a POC showing abilities, advantages and flaws of current SotA technologies applied to sports analytics.
        
        """
        st.markdown(disclaimer_text)
        
    with st.expander("***License***"):
        license_text = """
                **License**
                
                This application and its associated work are released under the **Creative Commons Attribution-NonCommercial 4.0 International License**.
                
                **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**

                You are free to:

                - **Share** - copy and redistribute the material in any medium or format.
                
                - **Adapt** - remix, transform, and build upon the material.

                Under the following terms:

                - **Attribution (BY)**: You must give appropriate credits and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

                - **Non-Commercial (NC)**: You may not use the material for commercial purposes.

                - **No Derivatives (ND)**: If you remix, transform, or build upon the material, you may not distribute the modified material.

                For a full description of the license terms, please visit the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

                This license is designed to allow others to use, remix, and build upon your work, but not for commercial purposes. It requires proper attribution and restricts commercial use and the creation of derivative works.
                """
        st.markdown(license_text)