import streamlit as st

from src.TokenFlow.run_tokenflow_sdedit import run

# Import other required modules

plugin_info = {"name": "Stable Diffusion Processor"}


def plugin_tab(*args, **kwargs):
    if "output_path" not in st.session_state:
        st.session_state.output_path = "output/path"
    if "n_timesteps" not in st.session_state:
        st.session_state.n_timesteps = 50
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

    st.title("Stable Diffusion Processor")

    st.session_state.output_path = st.text_input("Output Path:", st.session_state.output_path)
    st.session_state.n_timesteps = st.number_input("Number of Time Steps:", value=st.session_state.n_timesteps)
    st.session_state.start = st.number_input("Start:", min_value=0.0, max_value=1.0, value=st.session_state.start)
    st.session_state.use_ddim_noise = st.checkbox("Use DDim Noise:", value=st.session_state.use_ddim_noise)
    st.session_state.prompt = st.text_input("Prompt:", st.session_state.prompt)
    st.session_state.negative_prompt = st.text_input("Negative Prompt:", st.session_state.negative_prompt)
    st.session_state.guidance_scale = st.number_input("Guidance Scale:", min_value=0.0,
                                                      value=st.session_state.guidance_scale)

    if st.button("Run Processing"):
        # Here, gather all the previous state variables, and any other required information
        config = {
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
            "batch_size": st.session_state.batch_size,
            "save_steps": st.session_state.save_steps,
            "n_frames": st.session_state.n_frames,
            "inversion_prompt": st.session_state.inversion_prompt,
        }
        # Call your main function, e.g., run_processing
        run(config)