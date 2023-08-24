import streamlit as st

plugin_info = {"name": "Doom"}

def plugin_tab(tabs, tab_names):
    select = st.selectbox('What', ['doom', 'auto', 'custom'])
    if select == 'doom':
        src = "https://www.retrogames.cz/play_414-DOS.php"
    elif select == 'auto':
        src = 'https://127.0.0.1:7860'
    else:
        src = st.text_input('Url')
    web = st.components.v1.iframe(src, width=None, height=1024, scrolling=True)
