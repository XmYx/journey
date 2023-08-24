# base webui import and utils.
import json
import random
import secrets
import time
from datetime import datetime, date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st
# from ui.sd_utils import *

# from scripts.tools.deforum_runner import runner
# from scripts.tools.nsp.nsp_pantry import parser

# streamlit imports
#from streamlit import StopException
#from streamlit.runtime.in_memory_file_manager import in_memory_file_manager
from streamlit.elements import image as STImage

# other imports

import os
from io import BytesIO

from deforum.animation.base_args import DeforumAnimPrompts
from deforum.animation.new_args import ParseqArgs, LoopArgs, DeforumArgs, DeforumAnimArgs, DeforumOutputArgs
from deforum.cmd import extract_values
from deforum.general_utils import substitute_placeholders

plugin_info = {"name": "Deforum"}

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging

    logging.set_verbosity_error()
except:
    pass
from omegaconf import OmegaConf

st.session_state["defaults"] = OmegaConf.load("config/streamlit.yaml")
class PluginInfo():
    plugname = "txt2vid"
    description = "Text to Video"
    isTab = True
    displayPriority = 1


if os.path.exists(os.path.join(st.session_state['defaults'].general.GFPGAN_dir, "experiments", "pretrained_models",
                               "GFPGANv1.3.pth")):
    GFPGAN_available = True
else:
    GFPGAN_available = False

if os.path.exists(os.path.join(st.session_state['defaults'].general.RealESRGAN_dir, "experiments", "pretrained_models",
                               f"{st.session_state['defaults'].txt2vid.RealESRGAN_model}.pth")):
    RealESRGAN_available = True
else:
    RealESRGAN_available = False


def get_output_folder(output_path, batch_folder):
    out_path = os.path.join(os.getcwd(), output_path, time.strftime('%Y-%m'), str(date.today().day))
    if batch_folder != "":
        out_path = os.path.join(out_path, batch_folder)
    print(f"Saving animation frames to {out_path}")
    os.makedirs(out_path, exist_ok=True)
    return out_path


def get_args():
    st.session_state["generation_mode"] = "txt2vid"
    # SimpleNamespace = type(sys.implementation)

    W, H = map(lambda x: x - x % 64,
               (st.session_state[st.session_state["generation_mode"]]['W'],
                st.session_state[st.session_state["generation_mode"]]['H']))  # resize to integer multiple of 64

    now = datetime.now()  # current date and time
    batch_name = now.strftime("%H_%M_%S")
    if st.session_state[st.session_state["generation_mode"]]["pathmode"] == "subfolders":
        out_folder = get_output_folder('./content/output', batch_name)
    else:
        out_folder = st.session_state[st.session_state["generation_mode"]]["outdir"]
    if st.session_state[st.session_state["generation_mode"]]['seed'] == '':
        st.session_state[st.session_state["generation_mode"]]['seed'] = int(random.randrange(0, 4200000000))
    else:
        st.session_state[st.session_state["generation_mode"]]['seed'] = int(
            st.session_state[st.session_state["generation_mode"]]['seed'])
    seed = int(random.randrange(0, 4200000000))
    args = {  # 'image': st.session_state[st.session_state["generation_mode"]]['preview_image'],
        # 'video': st.session_state[st.session_state["generation_mode"]]['preview_video'],
        'W': W,
        'H': H,
        'seed': st.session_state[st.session_state["generation_mode"]]['seed'],  # @param
        'sampler': st.session_state[st.session_state["generation_mode"]]['sampler'],
        # @param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
        'steps': st.session_state[st.session_state["generation_mode"]]['steps'],  # @param
        'scale': st.session_state[st.session_state["generation_mode"]]['scale'],  # @param
        'ddim_eta': st.session_state[st.session_state["generation_mode"]]['ddim_eta'],  # @param
        'dynamic_threshold': None,
        'static_threshold': None,

        # @markdown **Save & Display Settings**
        'save_samples': st.session_state[st.session_state["generation_mode"]]['save_samples'],
        # @param {type:"boolean"}
        'save_settings': st.session_state[st.session_state["generation_mode"]]['save_settings'],
        # @param {type:"boolean"}
        'display_samples': st.session_state[st.session_state["generation_mode"]]['display_samples'],
        # @param {type:"boolean"}

        # @markdown **Batch Settings**
        'n_batch': st.session_state[st.session_state["generation_mode"]]["iterations"],  # @param
        'batch_name': batch_name,  # @param {type:"string"}
        'filename_format': st.session_state[st.session_state["generation_mode"]]['filename_format'],
        # @param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
        'seed_behavior': st.session_state[st.session_state["generation_mode"]]['seed_behavior'],
        # @param ["iter","fixed","random"]
        'make_grid': st.session_state[st.session_state["generation_mode"]]['make_grid'],  # @param {type:"boolean"}
        'grid_rows': st.session_state[st.session_state["generation_mode"]]['grid_rows'],  # @param
        'outdir': out_folder,

        # @markdown **Init Settings**
        'use_init': st.session_state[st.session_state["generation_mode"]]['use_init'],  # @param {type:"boolean"}
        'strength': st.session_state[st.session_state["generation_mode"]]['strength'],  # @param {type:"number"}
        'strength_0_no_init': st.session_state[st.session_state["generation_mode"]]['strength_0_no_init'],
        # Set the strength to 0 automatically when no init image is used
        'init_image': st.session_state[st.session_state["generation_mode"]]['init_image'],  # @param {type:"string"}
        # Whiter areas of the mask are areas that change more
        'use_mask': st.session_state[st.session_state["generation_mode"]]['use_mask'],  # @param {type:"boolean"}
        'use_alpha_as_mask': st.session_state[st.session_state["generation_mode"]]['use_alpha_as_mask'],
        # use the alpha channel of the init image as the mask
        'mask_file': st.session_state[st.session_state["generation_mode"]]['mask_file'],  # @param {type:"string"}
        'invert_mask': st.session_state[st.session_state["generation_mode"]]['invert_mask'],  # @param {type:"boolean"}
        # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
        'mask_brightness_adjust': st.session_state[st.session_state["generation_mode"]]['mask_brightness_adjust'],
        # @param {type:"number"}
        'mask_contrast_adjust': st.session_state[st.session_state["generation_mode"]]['mask_contrast_adjust'],
        # @param {type:"number"}

        'n_samples': st.session_state[st.session_state["generation_mode"]]["batch_size"],
        'precision': 'autocast',
        'C': 4,
        'f': 8,

        'keyframes': st.session_state[st.session_state["generation_mode"]]['keyframes'],
        'prompt': st.session_state[st.session_state["generation_mode"]]['prompt'],

        'timestring': "",
        'init_latent': None,
        'init_sample': None,
        'init_c': None,
    }

    if st.session_state["generation_mode"] == "txt2img":
        anim_args = {
            'animation_mode': None

        }
    else:

        anim_args = {'animation_mode': st.session_state[st.session_state["generation_mode"]]['animation_mode'],
                           # @param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
                           'max_frames': st.session_state[st.session_state["generation_mode"]]['max_frames'],
                           # @param {type:"number"}
                           'border': st.session_state[st.session_state["generation_mode"]]['border'],
                           # @param ['wrap', 'replicate'] {type:'string'}

                           # @markdown ####**Motion Parameters:**
                           'angle': st.session_state[st.session_state["generation_mode"]]['angle'],
                           # @param {type:"string"}
                           'zoom': st.session_state[st.session_state["generation_mode"]]['zoom'],
                           # @param {type:"string"}
                           'translation_x': st.session_state[st.session_state["generation_mode"]]['translation_x'],
                           # @param {type:"string"}
                           'translation_y': st.session_state[st.session_state["generation_mode"]]['translation_y'],
                           # @param {type:"string"}
                           'translation_z': st.session_state[st.session_state["generation_mode"]]['translation_z'],
                           # @param {type:"string"}
                           'rotation_3d_x': st.session_state[st.session_state["generation_mode"]]['rotation_3d_x'],
                           # @param {type:"string"}
                           'rotation_3d_y': st.session_state[st.session_state["generation_mode"]]['rotation_3d_y'],
                           # @param {type:"string"}
                           'rotation_3d_z': st.session_state[st.session_state["generation_mode"]]['rotation_3d_z'],
                           # @param {type:"string"}
                           'noise_schedule': st.session_state[st.session_state["generation_mode"]]['noise_schedule'],
                           # @param {type:"string"}
                           'flip_2d_perspective': st.session_state[st.session_state["generation_mode"]][
                               'flip_2d_perspective'],  # @param {type:"boolean"}
                           'perspective_flip_theta': st.session_state[st.session_state["generation_mode"]][
                               'perspective_flip_theta'],
                           # @param {type:"string"}
                           'perspective_flip_phi': st.session_state[st.session_state["generation_mode"]][
                               'perspective_flip_phi'],  # @param {type:"string"}
                           'perspective_flip_gamma': st.session_state[st.session_state["generation_mode"]][
                               'perspective_flip_gamma'],
                           # @param {type:"string"}
                           'perspective_flip_fv': st.session_state[st.session_state["generation_mode"]][
                               'perspective_flip_fv'],  # @param {type:"string"}
                           'strength_schedule': st.session_state[st.session_state["generation_mode"]][
                               'strength_schedule'],  # @param {type:"string"}
                           'contrast_schedule': st.session_state[st.session_state["generation_mode"]][
                               'contrast_schedule'],  # @param {type:"string"}

                           # @markdown ####**Coherence:**
                           'color_coherence': st.session_state[st.session_state["generation_mode"]]['color_coherence'],
                           # @param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
                           'diffusion_cadence': st.session_state[st.session_state["generation_mode"]][
                               'diffusion_cadence'],
                           # @param ['1','2','3','4','5','6','7','8'] {type:'string'}

                           # @markdown ####**3D Depth Warping:**
                           'use_depth_warping': st.session_state[st.session_state["generation_mode"]][
                               'use_depth_warping'],  # @param {type:"boolean"}
                           'midas_weight': st.session_state[st.session_state["generation_mode"]]['midas_weight'],
                           # @param {type:"number"}
                           'near_plane': st.session_state[st.session_state["generation_mode"]]['near_plane'],
                           'far_plane': st.session_state[st.session_state["generation_mode"]]['far_plane'],
                           'fov': st.session_state[st.session_state["generation_mode"]]['fov'],
                           # @param {type:"number"}
                           'padding_mode': st.session_state[st.session_state["generation_mode"]]['padding_mode'],
                           # @param ['border', 'reflection', 'zeros'] {type:'string'}
                           'sampling_mode': st.session_state[st.session_state["generation_mode"]]['sampling_mode'],
                           # @param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
                           'save_depth_maps': st.session_state[st.session_state["generation_mode"]]['save_depth_maps'],
                           # @param {type:"boolean"}

                           # @markdown ####**Video Input:**
                           'video_init_path': st.session_state[st.session_state["generation_mode"]]['video_init_path'],
                           # @param {type:"string"}
                           'extract_nth_frame': st.session_state[st.session_state["generation_mode"]][
                               'extract_nth_frame'],  # @param {type:"number"}

                           # @markdown ####**Interpolation:**
                           'interpolate_key_frames': st.session_state[st.session_state["generation_mode"]][
                               'interpolate_key_frames'],
                           # @param {type:"boolean"}
                           'interpolate_x_frames': st.session_state[st.session_state["generation_mode"]][
                               'interpolate_x_frames'],  # @param {type:"number"}

                           # @markdown ####**Resume Animation:**
                           'resume_from_timestring': st.session_state[st.session_state["generation_mode"]][
                               'resume_from_timestring'],
                           # @param {type:"boolean"}
                           'resume_timestring': st.session_state[st.session_state["generation_mode"]][
                               'resume_timestring']  # @param {type:"string"}

                           }

    # args = SimpleNamespace(**DeforumArgs)
    # anim_args = SimpleNamespace(**DeforumAnimArgs)

    return args, anim_args


#
def plugin_tab(tabs, tab_names):
    #def_runner = runner()

    with st.form("txt2vid-deforum"):
        st.session_state["generation_mode"] = "txt2vid"
        st.session_state["txt2vid"] = {}

        # input_col1, generate_col1 = st.columns([10,1])
        # with input_col1:
        # prompt = st.text_area("Input Text","")

        # Every form must have a submit button, the extra blank spaces is a temp way to align it with the input field. Needs to be done in CSS or some other way.
        # generate_col1.write("")
        # generate_col1.write("")

        # creating the page layout using columns
        col1, col2, col3 = st.columns([1, 2, 1], gap="small")
        with col1:
            with st.expander("Basic Settings", expanded=True):
                generate_button = st.form_submit_button("Generate")

                st.session_state["txt2vid"]["prompt"] = st.text_area("Input Text", value='',
                                                                     placeholder="A corgi wearing a top hat.\nSecond Prompt")
                st.session_state["txt2vid"]["keyframes"] = st.text_area("Keyframes", "0", placeholder="0\n5\n10")

                st.session_state["txt2vid"]["max_frames"] = st.slider("Max Frames:", min_value=1, max_value=2048,
                                                                      value=st.session_state['defaults'].txt2vid.max_frames, step=1)

                st.session_state["txt2vid"]["W"] = st.slider("Width:", min_value=64, max_value=2048,
                                                             value=st.session_state['defaults'].txt2vid.W, step=64)
                st.session_state["txt2vid"]["H"] = st.slider("Height:", min_value=64, max_value=2048,
                                                             value=st.session_state['defaults'].txt2vid.H, step=64)
                st.session_state["txt2vid"]["scale"] = st.slider("CFG (Classifier Free Guidance Scale):", min_value=1.0,
                                                                 max_value=30.0, value=st.session_state['defaults'].txt2vid.scale,
                                                                 step=1e-1, format="%.1f",
                                                                 help="How strongly the image should follow the prompt.")

        # with st.expander(""):
        with col2:
            preview_tab, prompt_tab, rendering_tab, settings_tab = st.tabs(["Preview",
                                                                            "Prompt help",
                                                                            "Rendering",
                                                                            "Settings"
                                                                            ])
            with prompt_tab:
                # nsp = parser()
                # nsp_keys = nsp.get_nsp_keys()

                # inputprompt = st.multiselect('Topics', nsp_keys, key='prompts_ms_2')
                # st.text_input(label="Prompt Sample", value=nsp.parse(inputprompt), key='prompt_helper')

                st.session_state["txt2vid"]["prompt_tmp"] = st.text_area("Park your samples here", value='')

            with preview_tab:
                # st.write("Image")
                # Image for testing
                # image = Image.open(requests.get("https://icon-library.com/images/image-placeholder-icon/image-placeholder-icon-13.jpg", stream=True).raw).convert('RGB')
                # new_image = image.resize((175, 240))
                # preview_image = st.image(image)

                # create an empty container for the image, progress bar, etc so we can update it later and use session_state to hold them globally.
                st.session_state["txt2vid"]["preview_image"] = st.empty()

                st.session_state["txt2vid"]["loading"] = st.empty()

                st.session_state["txt2vid"]["progress_bar_text"] = st.empty()
                st.session_state["txt2vid"]["progress_bar"] = st.empty()

                # generate_video = st.empty()
                if "mp4_path" not in st.session_state:
                    st.session_state["txt2vid"]["preview_video"] = st.empty()
                else:
                    st.session_state["txt2vid"]["preview_video"] = st.video(st.session_state["mp4_path"])

                message = st.empty()
            with rendering_tab:
                sampler_tab, sequence_tab, flip_sequence_tab, frames_tab = st.tabs(["Sampler",
                                                                                    "3D Animation Sequence",
                                                                                    "2D Flip Sequence",
                                                                                    "Frame Setup"
                                                                                    ])

                with sampler_tab:

                    st.session_state["txt2vid"]["steps"] = st.number_input('Sample Steps',
                                                                           value=st.session_state['defaults'].txt2vid.steps, step=1)
                    st.session_state["txt2vid"]["sampler"] = st.selectbox(
                        'Sampler',
                        ("ddim", "plms", "klms", "dpm2", "dpm2_ancestral", "heun", "euler", "euler_ancestral"),
                        help="DDIM and PLMS are for quick results, you can use low sample steps. for the rest go up with the steps maybe start at 50 and raise from there")

                    st.session_state["txt2vid"]["sampling_mode"] = st.selectbox(
                        'Sampling Mode',
                        ('bicubic', 'bilinear', 'nearest'))
                    st.session_state["txt2vid"]["seed"] = st.text_input("Seed:", value=st.session_state['defaults'].txt2vid.seed,
                                                                        help=" The seed to use, if left blank a random seed will be generated.")
                    st.session_state["txt2vid"]["seed_behavior"] = st.selectbox(
                        'Seed Behavior',
                        ("iter", "fixed", "random"))
                with sequence_tab:
                    # col4, col5 = st.columns([1,1], gap="medium")
                    st.session_state["txt2vid"]["angle"] = st.text_input("Angle:", value=st.session_state['defaults'].txt2vid.angle)
                    st.session_state["txt2vid"]["zoom"] = st.text_input("Zoom:", value=st.session_state['defaults'].txt2vid.zoom)
                    st.session_state["txt2vid"]["translation_x"] = st.text_input("X Translation:",
                                                                                 value=st.session_state['defaults'].txt2vid.translation_x)
                    st.session_state["txt2vid"]["translation_y"] = st.text_input("Y Translation:",
                                                                                 value=st.session_state['defaults'].txt2vid.translation_y)
                    st.session_state["txt2vid"]["translation_z"] = st.text_input("Z Translation:",
                                                                                 value=st.session_state['defaults'].txt2vid.translation_z)
                    st.session_state["txt2vid"]["rotation_3d_x"] = st.text_input("X 3D Rotaion:",
                                                                                 value=st.session_state['defaults'].txt2vid.rotation_3d_x)
                    st.session_state["txt2vid"]["rotation_3d_y"] = st.text_input("Y 3D Rotaion:",
                                                                                 value=st.session_state['defaults'].txt2vid.rotation_3d_y)
                    st.session_state["txt2vid"]["rotation_3d_z"] = st.text_input("Z 3D Rotaion:",
                                                                                 value=st.session_state['defaults'].txt2vid.rotation_3d_z)
                    st.session_state["txt2vid"]["noise_schedule"] = st.text_input("Noise Schedule:", value=st.session_state[
                        'defaults'].txt2vid.noise_schedule)
                    st.session_state["txt2vid"]["strength_schedule"] = st.text_input("Strength Schedule:", value=st.session_state[
                        'defaults'].txt2vid.strength_schedule)
                    st.session_state["txt2vid"]["contrast_schedule"] = st.text_input("Contrast Schedule:", value=st.session_state[
                        'defaults'].txt2vid.contrast_schedule)
                with flip_sequence_tab:
                    st.session_state["txt2vid"]["flip_2d_perspective"] = st.checkbox('Flip 2d Perspective', value=False)
                    st.session_state["txt2vid"]["perspective_flip_theta"] = st.text_input("Flip Theta:", value=st.session_state[
                        'defaults'].txt2vid.perspective_flip_theta)
                    st.session_state["txt2vid"]["perspective_flip_phi"] = st.text_input("Flip Phi:", value=st.session_state[
                        'defaults'].txt2vid.perspective_flip_phi)
                    st.session_state["txt2vid"]["perspective_flip_gamma"] = st.text_input("Flip Gamma:", value=st.session_state[
                        'defaults'].txt2vid.perspective_flip_gamma)
                    st.session_state["txt2vid"]["perspective_flip_fv"] = st.text_input("Flip FV:", value=st.session_state[
                        'defaults'].txt2vid.perspective_flip_fv)
                with frames_tab:
                    basic_tab, mask_tab, init_tab = st.tabs(["Basics", "Mask", "Init Image"])
                    with basic_tab:
                        st.session_state["txt2vid"]["ddim_eta"] = st.number_input('DDIM ETA',
                                                                                  value=st.session_state['defaults'].txt2vid.ddim_eta,
                                                                                  step=1e-1, format="%.1f")

                        st.session_state["txt2vid"]["make_grid"] = st.checkbox('Make Grid', value=False)
                        st.session_state["txt2vid"]["grid_rows"] = st.number_input('Hight',
                                                                                   value=st.session_state['defaults'].txt2vid.grid_rows,
                                                                                   step=1)

                    with mask_tab:
                        st.session_state["txt2vid"]["use_mask"] = st.checkbox('Use Mask', value=False)
                        st.session_state["txt2vid"]["use_alpha_as_mask"] = st.checkbox('Use Alpha as Mask', value=False)
                        st.session_state["txt2vid"]["mask_file"] = st.text_input("Init Image:",
                                                                                 value=st.session_state['defaults'].txt2vid.mask_file,
                                                                                 help="The Mask to be used")
                        st.session_state["txt2vid"]["invert_mask"] = st.checkbox('Invert Mask', value=False)
                        st.session_state["txt2vid"]["mask_brightness_adjust"] = st.number_input('Brightness Adjust',
                                                                                                value=st.session_state[
                                                                                                    'defaults'].txt2vid.mask_brightness_adjust,
                                                                                                step=1e-1, format="%.1f",
                                                                                                help="Adjust the brightness of the mask")
                        st.session_state["txt2vid"]["mask_contrast_adjust"] = st.number_input('Contrast Adjust', value=st.session_state[
                            'defaults'].txt2vid.mask_contrast_adjust, step=1e-1, format="%.1f",
                                                                                              help="Adjust the contrast of the mask")
                    with init_tab:
                        st.session_state["txt2vid"]["use_init"] = st.checkbox('Use Init', value=False)
                        st.session_state["txt2vid"]["strength"] = st.number_input('Strength',
                                                                                  value=st.session_state['defaults'].txt2vid.strength,
                                                                                  step=1e-1, format="%.1f")
                        st.session_state["txt2vid"]["strength_0_no_init"] = st.checkbox('Strength 0', value=True,
                                                                                        help="Set the strength to 0 automatically when no init image is used")
                        st.session_state["txt2vid"]["init_image"] = st.text_input("Init Image:",
                                                                                  value=st.session_state['defaults'].txt2vid.init_image,
                                                                                  help="The image to be used as init")
            with settings_tab:
                st.session_state["txt2vid"]["save_samples"] = st.checkbox('Save Samples', value=True)
                st.session_state["txt2vid"]["save_settings"] = st.checkbox('Save Settings', value=False)  # For now
                st.session_state["txt2vid"]["display_samples"] = st.checkbox('Display Samples', value=True)
                st.session_state["txt2vid"]["pathmode"] = st.selectbox('Path Structure', ("subfolders", "root"),
                                                                       index=st.session_state[
                                                                           'defaults'].general.default_path_mode_index,
                                                                       help="subfolders structure will create daily folders plus many subfolders, root will use your outdir as root",
                                                                       key='pathmode-txt2vid')
                st.session_state["txt2vid"]["outdir"] = st.text_input("Output Folder",
                                                                      value=st.session_state['defaults'].general.outdir,
                                                                      help=" Output folder", key='outdir-txt2vid')
                st.session_state["txt2vid"]["filename_format"] = st.selectbox(
                    'Filename Format',
                    ("{timestring}_{index}_{seed}.png", "{timestring}_{index}_{prompt}.png"))

        with col3:
            # If we have custom models available on the "models/custom"
            # folder then we show a menu to select which model we want to use, otherwise we use the main model for SD
            # if st.session_state["CustomModel_available"]:
            #    custom_model = st.selectbox("Custom Model:", st.session_state["defaults"].txt2vid.custom_models_list,
            #                                index=st.session_state["defaults"].txt2vid.custom_models_list.index(st.session_state["defaults"].txt2vid.default_model),
            #                                help="Select the model you want to use. This option is only available if you have custom models \
            #                        on your 'models/custom' folder. The model name that will be shown here is the same as the name\
            #                        the file for the model has on said folder, it is recommended to give the .ckpt file a name that \
            #                    will make it easier for you to distinguish it from other models. Default: Stable Diffusion v1.4")
            # else:
            #    custom_model = "CompVis/stable-diffusion-v1-4"

            # st.session_state["weights_path"] = custom_model
            # else:
            # custom_model = "CompVis/stable-diffusion-v1-4"
            # st.session_state["weights_path"] = f"CompVis/{slugify(custom_model.lower())}"

            # basic_tab, advanced_tab = st.tabs(["Basic", "Advanced"])

            # with basic_tab:
            # summit_on_enter = st.radio("Submit on enter?", ("Yes", "No"), horizontal=True,
            # help="Press the Enter key to summit, when 'No' is selected you can use the Enter key to write multiple lines.")

            with st.expander("Animation", expanded=True):
                st.session_state["txt2vid"]["animation_mode"] = st.selectbox(
                    'Animation Mode',
                    ('2D', '3D'))
                st.session_state["txt2vid"]["border"] = st.selectbox(
                    'Border',
                    ('wrap', 'replicate'))
                st.session_state["txt2vid"]["color_coherence"] = st.selectbox(
                    'Color Coherence',
                    ('Match Frame 0 LAB', 'None', 'Match Frame 0 HSV', 'Match Frame 0 RGB'))
                st.session_state["txt2vid"]["diffusion_cadence"] = st.selectbox(
                    'Diffusion Cadence',
                    ('1', '2', '3', '4', '5', '6', '7', '8'))
                st.session_state["txt2vid"]["use_depth_warping"] = st.checkbox('Use Depth Warping', value=True)
                st.session_state["txt2vid"]["midas_weight"] = st.number_input('Midas Weight',
                                                                              value=st.session_state['defaults'].txt2vid.midas_weight,
                                                                              step=1e-1,
                                                                              format="%.1f")
                st.session_state["txt2vid"]["near_plane"] = st.number_input('Near Plane',
                                                                            value=st.session_state['defaults'].txt2vid.near_plane,
                                                                            step=1)
                st.session_state["txt2vid"]["far_plane"] = st.number_input('Far Plane',
                                                                           value=st.session_state['defaults'].txt2vid.far_plane,
                                                                           step=1)
                st.session_state["txt2vid"]["fov"] = st.number_input('FOV', value=st.session_state['defaults'].txt2vid.fov, step=1)
                st.session_state["txt2vid"]["padding_mode"] = st.selectbox(
                    'Padding Mode',
                    ('border', 'reflection', 'zeros'))

                st.session_state["txt2vid"]["save_depth_maps"] = st.checkbox('Save Depth Maps', value=False)
                st.session_state["txt2vid"]["video_init_path"] = ''
                st.session_state["txt2vid"]["extract_nth_frame"] = st.number_input('Extract Nth Frame', value=st.session_state[
                    'defaults'].txt2vid.extract_nth_frame, step=1)
                st.session_state["txt2vid"]["interpolate_key_frames"] = st.checkbox('Interpolate Key Frames', value=False)
                st.session_state["txt2vid"]["interpolate_x_frames"] = st.number_input('Number Frames to Interpolate',
                                                                                      value=st.session_state[
                                                                                          'defaults'].txt2vid.interpolate_x_frames,
                                                                                      step=1)
                st.session_state["txt2vid"]["resume_from_timestring"] = st.checkbox('Resume From Timestring', value=False)
                st.session_state["txt2vid"]["resume_timestring"] = st.text_input("Resume Timestring:", value=st.session_state[
                    'defaults'].txt2vid.resume_timestring, help="Some Video Path")
                st.session_state["txt2vid"]["iterations"] = 1
                st.session_state["txt2vid"]["batch_size"] = 1
                if generate_button:
                    from types import SimpleNamespace
                    from deforum.animation.new_args import RootArgs
                    from deforum.main import Deforum



                    args = extract_values(DeforumArgs())
                    anim_args = extract_values(DeforumAnimArgs())
                    ui_args, ui_anim_args = get_args()

                    args.update(ui_args)
                    anim_args.update(ui_anim_args)

                    args = SimpleNamespace(**args)
                    anim_args = SimpleNamespace(**anim_args)
                    print(args.prompt)

                    # expand prompts out to per-frame
                    prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames)])

                    if args.keyframes == '':
                        args.keyframes = "0"
                    prom = args.prompt
                    key = args.keyframes

                    new_prom = list(prom.split("\n"))
                    new_key = list(key.split("\n"))

                    prompts = dict(zip(new_key, new_prom))

                    # for i in animation_prompts:
                    #    prompt_series[i] = animation_prompts
                    #    print(prompt_series[i])

                    for i, prompt in prompts.items():
                        n = int(i)
                        prompt_series[n] = prompt
                    prompt_series = prompt_series.ffill().bfill()

                    print(prompt_series)
                    parseg_args = SimpleNamespace(**extract_values(ParseqArgs()))
                    loop_args = SimpleNamespace(**extract_values(LoopArgs()))
                    root = SimpleNamespace(**RootArgs())
                    video_args = SimpleNamespace(**extract_values(DeforumOutputArgs()))
                    controlnet_args = None
                    deforum = Deforum(args, anim_args, video_args, parseg_args, loop_args, controlnet_args, root)
                    setattr(deforum.loop_args, "init_images", "")
                    #animation_prompts = DeforumAnimPrompts()
                    #deforum.root.animation_prompts = json.loads(animation_prompts)
                    #deforum.animation_prompts = deforum.root.animation_prompts

                    deforum.root.animation_prompts = prompt_series
                    deforum.animation_prompts = prompt_series


                    deforum.args.timestring = time.strftime('%Y%m%d%H%M%S')
                    current_arg_list = [deforum.args, deforum.anim_args, deforum.video_args, deforum.parseq_args]
                    full_base_folder_path = os.path.join(os.getcwd(), "output/deforum")
                    deforum.root.raw_batch_name = deforum.args.batch_name
                    deforum.args.batch_name = substitute_placeholders(deforum.args.batch_name, current_arg_list,
                                                                      full_base_folder_path)
                    deforum.args.outdir = os.path.join(full_base_folder_path, str(deforum.args.batch_name))
                    if deforum.args.seed == -1 or deforum.args.seed == "-1":
                        setattr(deforum.args, "seed", secrets.randbelow(999999999999999999))
                        setattr(deforum.root, "raw_seed", int(deforum.args.seed))
                        setattr(deforum.args, "seed_internal", 0)
                    else:
                        deforum.args.seed = int(deforum.args.seed)
                    success = deforum()

                    print("deforum ran successfully")
