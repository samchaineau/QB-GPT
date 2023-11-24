import streamlit as st
from pages import set_app_title_and_logo, qb_gpt_page, contacts_and_disclaimers
import json
import pandas as pd
import numpy as np
import os
from tools import tokenizer

from assets.models import QBGPT

moves_to_pred = 11170
input_size = 11172
starts_size = 1954
scrimmage_size = 100
positions_id = 29

temp_ids = 52
off_def_size = 2
token_type_size = 3
play_type_size = 9

qbgpt = QBGPT(input_vocab_size = input_size,
                    positional_vocab_size = temp_ids,
                    position_vocab_size=positions_id,
                    start_vocab_size=starts_size,
                    scrimmage_vocab_size=scrimmage_size,
                    offdef_vocab_size = off_def_size,
                    type_vocab_size = token_type_size,
                    playtype_vocab_size = play_type_size,
                    embedding_dim = 256,
                    hidden_dim = 256,
                    num_heads = 3,
                    diag_masks = False,
                    to_pred_size = moves_to_pred)

qbgpt.load_weights("app/assets/model_mediumv2/QBGPT")


qb_tok = tokenizer(moves_index="./app/assets/moves_index.parquet",
                   play_index="./app/assets/plays_index.parquet",
                   positions_index="./app/assets/positions_index.parquet",
                   scrimmage_index="./app/assets/scrimmage_index.parquet",
                   starts_index="./app/assets/starts_index.parquet",
                   time_index="./app/assets/time_index.parquet",
                   window_size=20)

print(os.listdir("app"))

with open('./app/assets/ref.json', 'r') as fp:
    ref_json = json.load(fp)
    
def convert_numpy(d):
    return {k:np.array(v) for k,v in d.items()}

ref_json = {int(k):convert_numpy(v) for k,v in ref_json.items()}

ref_df = pd.read_json("./app/assets/ref_df.json")



# Define the main function to run the app
def main():
    set_app_title_and_logo()
    
    # Create a sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ("QB-GPT", "Contacts and Disclaimers"))

    if page == "QB-GPT":
        # Page 2: QB-GPT
        st.title("QB-GPT")
        qb_gpt_page(ref_df, ref_json, qb_tok, qbgpt)
        
    if page == "Contacts and Disclaimers":
        contacts_and_disclaimers()
        
        
if __name__ == "__main__":
    main()