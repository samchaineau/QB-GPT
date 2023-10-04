import streamlit as st
import plotly.graph_objs as go
import numpy as np
from PIL import Image

def set_app_title_and_logo():
    st.set_page_config(
        page_title="QB-GPT",
        page_icon=":rocket:",
        layout="wide",
    )
    
def header_menu():
    header_container = st.container()
    header_columns = header_container.columns(4)  # Create 4 columns for the buttons

    # Add navigation buttons within the header columns
    if header_columns[0].button("About", use_container_width=True):
        page = "About"
    elif header_columns[1].button("QB-GPT", use_container_width=True):
        page = "QB-GPT"
    elif header_columns[2].button("Helenos", use_container_width=True):
        page = "Helenos"
    elif header_columns[3].button("Contacts and Disclaimers", use_container_width=True):
        page = "Contacts and Disclaimers"
    else:
        page = "About"    # Set a default page when no button is clicked
    
def qb_gpt_page():

    # Scrimmage Line Input
    colinput1, colinput2, colinput3 = st.columns([2, 4, 4])
    
    with colinput1:
        with st.expander("Scrimmage Line"):
            scrimmage_line = st.number_input("Scrimmage Line (1-99)", min_value=1, max_value=99, value = 50)
    
    with colinput2:
        if st.button("Create random play", key="random_play_button", help="Click to create a random play.", use_container_width=True):
            # Add your action for the "Create random play" button here
            st.write("Create random play button clicked!")
        
    with colinput3:
        # Add a button in the bottom left corner of the plot
        if st.button("Generate", key="generate_button", help="Click to generate something.", use_container_width=True):
            # Add your action for the "Generate" button here
            st.write("Generate button clicked!")
            
    col1, col2 = st.columns(2)
    
    players = []

    # Defense Inputs
    with col1:
        st.header("Defense")
        with st.expander("Manually define Defense"):
            play_type_defense = st.selectbox(f"Play Type Defense", ["Choice 1", "Choice 2", "Choice 3"], key = "playtype_def")
            for i in range(11):
                st.subheader(f"Player {i+1}")
                position_x_y_defense = st.columns(3)
                with position_x_y_defense[0]:
                    position_key_def = "Position Defense"+ str(i)
                    position_defense = st.selectbox("Position", ["Position 1", "Position 2", "Position 3"], key = position_key_def)
                with position_x_y_defense[1]:
                    x_key_def = "x Defense"+ str(i)
                    x_defense = st.number_input("Starting X", min_value=0.0, max_value=100.0, step=0.01, format="%f", key = x_key_def)
                with position_x_y_defense[2]:
                    y_key_def = "y Defense"+ str(i)
                    y_defense = st.number_input("Starting Y", min_value=0.0, max_value=100.0, step=0.01, format="%f", key = y_key_def)
                players.append({
                    'position': position_defense,
                    'x': x_defense,
                    'y': y_defense,
                    'team' : "defense"
                })
    
    with col2:
        st.header("Offense")
        with st.expander("Manually define Offense"):
            play_type_offense = st.selectbox(f"Play Type Offense", ["Choice 1", "Choice 2", "Choice 3"], key = "playtype_off")
            for i in range(11):
                st.subheader(f"Player {i+1}")
                position_x_y_offense = st.columns(3)
                with position_x_y_offense[0]:
                    position_key_off = "Position Offense"+ str(i)
                    position_offense = st.selectbox("Position", ["Position 1", "Position 2", "Position 3"], key = position_key_off)
                with position_x_y_offense[1]:
                    x_key_off = "x Offense"+ str(i)
                    x_offense = st.number_input("Starting X", min_value=0.0, max_value=100.0, step=0.01, format="%f", key = x_key_off)
                with position_x_y_offense[2]:
                    y_key_off = "y Offense"+ str(i)
                    y_offense = st.number_input("Starting Y", min_value=0.0, max_value=100.0, step=0.01, format="%f", key = y_key_off)
                players.append({
                    'position': position_offense,
                    'x': x_offense,
                    'y': y_offense,
                    'team' : "offense"
                })
                    
    fig = go.Figure()

    # Add vertical line at scrimmage line
    fig.add_shape(
        go.layout.Shape(
            type="line",
            x0=scrimmage_line,
            x1=scrimmage_line,
            y0=0,
            y1=100,
            line=dict(color="black", width=2)
        )
    )
    

    # Add dots for players with different colors for offense and defense
    for player_data in players:
        if player_data['team'] == "defense":
            color = "blue"
        elif player_data['team'] == "offense":
            color = "red"
            
        fig.add_trace(
            go.Scatter(
                x=[player_data['x']],
                y=[player_data['y']],
                mode="markers",
                marker=dict(size=10, color=color),
                name=f"Position: {player_data['position']}"
            )
        )

    fig.update_layout(
    width=1200,
    height=600,
    title="Scrimmage Line and Player Positions",
    title_font=dict(color="black"),
    xaxis=dict(title="X Coordinate", range=[-10, 110]),
    yaxis=dict(title="Y Coordinate", range=[0, 60]),
    paper_bgcolor="grey",  # Set background color to grey
    plot_bgcolor="grey",  # Set plot background color to grey
    xaxis_showgrid=False,  # Remove x-axis grid lines
    yaxis_showgrid=False,  # Remove y-axis grid lines
    bargap=0.05  # Adjust the gap between bars
    )

    st.plotly_chart(fig)
    
def select_players(team_type, team_name):
    players = []
    st.header(team_type)
    with st.expander(f"Select {team_type} Players"):
        for i in range(11):
            st.subheader(f"Player {i + 1} - {team_name}")
            choices = st.columns(1)
            with choices[0]:
                choice_key = f"Position {team_type} {i}"
                choice = st.selectbox("Name, position and jersey number", ["Position 1", "Position 2", "Position 3"], key=choice_key)

            players.append({
                'position': choice,
                'team': team_name
            })
    return players
    
def helenos_page():
    col1, col2, col3 = st.columns([2, 2, 1])
    # Input for selecting teams
    
    with col1:
        with st.expander("Select Teams"):
            offense_team = st.text_input("Offense Team Name", "Offense Team")
            defense_team = st.text_input("Defense Team Name", "Defense Team")

    # Input for down and season
    with col2:
        with st.expander("Game Info"):
            down = st.number_input("Down (1-4)", min_value=1, max_value=4, value=1)
            season = st.number_input("Season (2017-2022)", min_value=2017, max_value=2022, value=2022)
            
    with col3:
        if st.button("Predict", key="prediction", help="Click to make a prediction.", use_container_width=True):
            # Add your action for the "Create random play" button here
            st.write("Make a prediction clicked!")
            
    col4, col5 = st.columns(2)
    # Player selection for Offense
    with col4:
        select_players("Offense", offense_team)

    # Player selection for Defense
    with col5:
        select_players("Defense", defense_team)
        
    data = np.random.normal(0, 10, 1000)
    
    col6, sep, col7 = st.columns([4, 1, 3])
    
    with col6:
        # Create a histogram plot
        fig = go.Figure(data=[go.Histogram(x=data)])
        fig.update_layout(
            xaxis=dict(title="X Coordinate", range=[-100, 100]),
            yaxis=dict(title="Count"),
            title="Yards gained predicted",
            title_font=dict(color="black"),
            paper_bgcolor="grey",
            plot_bgcolor="grey",
            xaxis_showgrid=False,
            yaxis_showgrid=False
        )

        st.plotly_chart(fig)
    
    with col7:
        success_prob = np.random.random()
        fail_prob = 1.0 - success_prob

        pie_fig = go.Figure(data=[go.Pie(labels=["Success", "Fail"], values=[success_prob, fail_prob])])
        pie_fig.update_layout(
            title="Success vs. Fail Probability",
            title_font=dict(color="black")
        )

        st.plotly_chart(pie_fig)


def about_page():
    # insert widget here
    
    st.title("StratAI - Your Football Playbook Powerhouse!")
    
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
        
    with st.expander("***Helenos: Predicting the outcome of Football Plays***"):
        
        qb_gpt_hel = """
        QB-GPT serves as the backbone of our project, Helenos. While QB-GPT excels at play design, Helenos takes it a step further. We combine the learned embeddings from QB-GPT with other critical variables to predict the outcomes of pass and run plays with unprecedented accuracy. It's a game-changer in football analytics. You can compare Helenos to QB-GPT as a sentiment analysis model build upon a Language Model, you first learn how words are jointly connected before understanding a more global sense such as sentiment.
        
        It's the ultimate tool for coaches and analysts, helping them make data-driven decisions that can change the course of a game.
        
        A more detailed blogpost about the model Helenos can be found [here](link)
        """
        st.markdown(qb_gpt_hel)
        
    with st.expander("***Join and contacts***"):
        
        qb_gpt_join = """
        Are you passionate about data science, an NFL enthusiast, or simply intrigued by the fusion of technology and sports strategy? There are countless ways to become a part of the QB-GPT and Helenos journey. Whether you're a data scientist eager to explore cutting-edge AI in sports, a football fan wanting to dive into strategic gameplay, or someone who sees the immense potential in this project, there's a place for you. Join us in shaping the future of sports strategy. To get involved, simply reach out and let's connect. 
        
        See the contacts in the "Contacts and Disclaimers" section.
        """
        st.markdown(qb_gpt_join)
        
        
def contacts_and_disclaimers():
    
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