import streamlit as st
import json

# This is where we will indicate that we will have a multipage app
st.set_page_config()

# Use our file structure
with open('../config/filepaths.json', 'r') as f:
    FPATHS = json.load(f)

st.image(FPATHS['images']['banner'])

st.title('Choose page on sidebar')