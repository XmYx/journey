import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path

import streamlit as st

from src.TokenFlow.preprocess import prep
from src.TokenFlow.util import save_video_frames

plugin_info = {"name": "TokenFlow Preprocessor"}


def get_config():
    config_data = {
        "data_path": st.session_state.data_path,
        "H": st.session_state.H,
        "W": st.session_state.W,
        "save_path": st.session_state.save_path,
        "sd_version": st.session_state.sd_version,
        "steps": st.session_state.steps,
        "batch_size": st.session_state.batch_size,
        "save_steps": st.session_state.save_steps,
        "n_frames": st.session_state.n_frames,
        "inversion_prompt": st.session_state.inversion_prompt
    }
    return config_data

def plugin_tab(*args, **kwargs):

    with st.form('TokenFlow Preprocessing'):

        # Initialize session state variables
        if "data_path" not in st.session_state:
            st.session_state.data_path = ""
        if "save_path" not in st.session_state:
            st.session_state.save_path = ""
        if "H" not in st.session_state:
            st.session_state.H = 512
        if "W" not in st.session_state:
            st.session_state.W = 512
        if "sd_version" not in st.session_state:
            st.session_state.sd_version = "2.1"
        if "steps" not in st.session_state:
            st.session_state.steps = 500
        if "batch_size" not in st.session_state:
            st.session_state.batch_size = 40
        if "save_steps" not in st.session_state:
            st.session_state.save_steps = 50
        if "n_frames" not in st.session_state:
            st.session_state.n_frames = 40
        if "inversion_prompt" not in st.session_state:
            st.session_state.inversion_prompt = "a woman running"

        # UI components
        # st.title("Stable Diffusion Preprocessor")

        col1, col2 = st.columns([1,1])
        with col1:
            st.session_state.data_path = st.text_input("Data Path:", st.session_state.data_path)
            st.session_state.save_path = st.text_input("Save Path:", st.session_state.save_path)
            st.session_state.inversion_prompt = st.text_input("Inversion Prompt:", st.session_state.inversion_prompt)
            # Configuration name entry
            config_name = st.text_input("Configuration Name:", value="default_config")

            st.session_state.sd_version = st.selectbox("SD Version:", ["1.5", "2.0", "2.1", "ControlNet", "depth"])
            # Save and Load Configuration buttons
            configs_path = "configs/tokenflow"
            os.makedirs(configs_path, exist_ok=True)
            config_files = [f.stem for f in Path(configs_path).glob('*.json')]

            selected_config = st.selectbox("Select a configuration to load:", config_files)

        with col2:
            st.session_state.H = st.number_input("Height:", value=st.session_state.H)
            st.session_state.W = st.number_input("Width:", value=st.session_state.W)
            st.session_state.steps = st.number_input("Steps:", value=st.session_state.steps)
            st.session_state.batch_size = st.number_input("Batch Size:", value=st.session_state.batch_size)
            st.session_state.save_steps = st.number_input("Save Steps:", value=st.session_state.save_steps)
            st.session_state.n_frames = st.number_input("Number of Frames:", value=st.session_state.n_frames)



        col1, col2, col3, col4 = st.columns([1,1,1,3])

        with col4:
            st.session_state.progress_par = st.progress(1)

        def callback(i):

            max_value = st.session_state.steps
            progress_fraction = i / max_value
            st.session_state.progress_par.progress(progress_fraction)

        with col1:
            if st.form_submit_button("Save Configuration"):
                config_data = get_config()
                with open(Path(configs_path) / f"{config_name}.json", 'w') as json_file:
                    json.dump(config_data, json_file)
        with col2:

            if st.form_submit_button("Load Configuration"):
                with open(Path(configs_path) / f"{selected_config}.json", 'r') as json_file:
                    loaded_config = json.load(json_file)
                    for key, value in loaded_config.items():
                        setattr(st.session_state, key, value)
                st.experimental_rerun()
        with col3:
            if st.form_submit_button("Run Preprocessing"):
                # Create an options object (this could be a dictionary or an argparse.Namespace)

                opt = argparse.Namespace(**get_config())

                video_name = Path(st.session_state.data_path).stem
                if not os.path.isdir(f'data/{video_name}'):
                    save_video_frames(st.session_state.data_path, img_size=(opt.H, opt.W))
                opt.data_path = os.path.join('data', Path(st.session_state.data_path).stem)
                os.makedirs(opt.data_path, exist_ok=True)
                opt.callback = callback
                prep(opt)