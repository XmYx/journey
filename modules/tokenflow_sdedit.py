import json
import os
import secrets
import sys
from pathlib import Path

import streamlit as st

from src.TokenFlow.run_tokenflow_sdedit import run as run_sdedit
from src.TokenFlow.run_tokenflow_pnp import run as run_pnpedit

# Import other required modules

plugin_info = {"name": "TokenFlow Processor"}


def get_edit_config():

    # pnp params -- injection thresholds ∈ [0, 1]
    pnp_attn_t = 0.5
    pnp_f_t = 0.8

    seed = st.session_state.seed if st.session_state.seed != -1 else secrets.randbelow(4294967296)
    print('seed', seed)
    config = {
        "pnp_attn_t": pnp_attn_t,
        "pnp_f_t": pnp_f_t,
        "output_path": st.session_state.output_path,
        "n_timesteps": st.session_state.n_timesteps,
        "start": st.session_state.start,
        "use_ddim_noise": st.session_state.use_ddim_noise,
        "prompt": st.session_state.prompt,
        "negative_prompt": st.session_state.negative_prompt,
        "guidance_scale": st.session_state.guidance_scale,
        # Add other parameters from preprocessing tab
        "data_path": st.session_state.data_path,
        "H": st.session_state.H,
        "W": st.session_state.W,
        "sd_version": st.session_state.sd_version,
        "steps": st.session_state.steps,
        "n_inversion_steps": st.session_state.inversion_steps,
        "batch_size": st.session_state.batch_size,
        "save_steps": st.session_state.save_steps,
        "n_frames": st.session_state.n_frames,
        "inversion_prompt": st.session_state.inversion_prompt,
        "seed": seed,
        "device": "cuda",
        "latents_path": st.session_state.save_path,
    }
    return config

def plugin_tab(*args, **kwargs):

    with st.form('TokenFlow Editing'):

        if "output_path" not in st.session_state:
            st.session_state.output_path = "output/path"
        if "save_path" not in st.session_state:
            st.session_state.save_path = ""
        if "n_timesteps" not in st.session_state:
            st.session_state.n_timesteps = 50
        if "inversion_steps" not in st.session_state:
            st.session_state.inversion_steps = 500
        if "start" not in st.session_state:
            st.session_state.start = 0.9
        if "use_ddim_noise" not in st.session_state:
            st.session_state.use_ddim_noise = True
        if "prompt" not in st.session_state:
            st.session_state.prompt = "a shiny silver robotic wolf"
        if "negative_prompt" not in st.session_state:
            st.session_state.negative_prompt = "ugly, blurry, low res, unrealistic, unaesthetic"
        if "guidance_scale" not in st.session_state:
            st.session_state.guidance_scale = 7.5
        if "seed" not in st.session_state:
            st.session_state.seed = -1
        col1, col2 = st.columns([5,5])

        with col1:
            method = st.selectbox('Method', ['PNP', 'SDEdit'])
            st.session_state.sd_version = st.selectbox("SD Version:", ["1.5", "2.0", "2.1", "ControlNet", "depth"])
            st.session_state.output_path = st.text_input("Output Path:", st.session_state.output_path)
            st.session_state.data_path = st.text_input("Data Path:", st.session_state.data_path)
            st.session_state.n_timesteps = st.number_input("Number of Time Steps:", value=st.session_state.n_timesteps)
            st.session_state.seed = st.number_input("Seed:", value=st.session_state.seed)
            st.session_state.start = st.number_input("Start:", min_value=0.0, max_value=1.0, value=st.session_state.start)
            st.session_state.save_path = st.text_input("Preprocessed Latent Path:", st.session_state.save_path)

        with col2:
            st.session_state.prompt = st.text_input("Prompt:", st.session_state.prompt)
            st.session_state.negative_prompt = st.text_input("Negative Prompt:", st.session_state.negative_prompt)
            st.session_state.guidance_scale = st.number_input("Guidance Scale:", min_value=0.0,
                                                              value=st.session_state.guidance_scale)
            st.session_state.inversion_steps = st.number_input("Inversion Steps:", value=st.session_state.inversion_steps)
            st.session_state.n_frames = st.number_input("Number of Frames:", value=st.session_state.n_frames)


            # Configuration name entry
            config_name = st.text_input("Configuration Name:", value="default_config")

            # Save and Load Configuration buttons
            configs_path = "configs/tokenflow_edit"
            os.makedirs(configs_path, exist_ok=True)
            config_files = [f.stem for f in Path(configs_path).glob('*.json')]

            selected_config = st.selectbox("Select a configuration to load:", config_files)
        a_col1, a_col2, a_col3, a_col4, a_col_5 = st.columns([1,1,1,1,1])
        with a_col1:
            if st.form_submit_button("Save Configuration"):
                config_data = get_edit_config()
                with open(Path(configs_path) / f"{config_name}.json", 'w') as json_file:
                    json.dump(config_data, json_file)
        with a_col2:
            if st.form_submit_button("Load Configuration"):
                with open(Path(configs_path) / f"{selected_config}.json", 'r') as json_file:
                    loaded_config = json.load(json_file)
                    for key, value in loaded_config.items():
                        setattr(st.session_state, key, value)
                st.experimental_rerun()
        with a_col3:
            if st.form_submit_button("Run Processing"):
                # Here, gather all the previous state variables, and any other required information

                config = get_edit_config()

                # Call your main function, e.g., run_processing
                if method == 'SDEdit':
                    run_sdedit(config)
                else:
                    run_pnpedit(config)
        with a_col4:
            st.session_state.use_ddim_noise = st.checkbox("Use DDim Noise:", value=st.session_state.use_ddim_noise)
