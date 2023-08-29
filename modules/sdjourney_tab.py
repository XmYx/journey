# Stable Diffusion XL / Midjourney experience
from collections import deque
import streamlit as st
import numpy as np
from PIL import Image
import secrets
from enum import Enum

import streamlit as st

import torch
from PIL import Image
from diffusers import (DiffusionPipeline, DDIMScheduler, HeunDiscreteScheduler, KDPM2DiscreteScheduler,
                       KDPM2AncestralDiscreteScheduler,
                       LMSDiscreteScheduler, PNDMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
                       DPMSolverMultistepScheduler,
                       AutoencoderTiny, DPMSolverSinglestepScheduler, StableDiffusionXLPipeline,
                       StableDiffusionXLImg2ImgPipeline, AutoPipelineForText2Image, KandinskyV22CombinedPipeline)

from extras.call import new_call

from extras.sdjourney_backend import  controls_config, get_generation_args, SchedulerType, get_scheduler

from extras.streamlit_helpers import dynamic_controls
from main import singleton as gs

if "models" not in gs.data:
    print("Instantiating models dictionary in singleton")
    gs.data["models"] = {}

plugin_info = {"name": "SDJourney"}

ON = True
OFF = False

lowvram = OFF

def generate(args, callback):
    target_device = "cuda"
    from modules.sdjourney_tab import lowvram

    if not lowvram:
        if gs.data["models"]["base"].device.type != target_device:
            gs.data["models"]["base"].to(target_device)
    args["callback"] = callback
    if isinstance(gs.data["models"]["base"], KandinskyV22CombinedPipeline):
        st.session_state["images"] = gs.data["models"]["base"](**args).images
        return st.session_state["images"]
    else:
        result = gs.data["models"]["base"].generate(**args)
    st.session_state.latents = result[0].clone().detach()
    callback(args["num_inference_steps"], None, None)

    st.session_state["images"] = result[1]

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


    return result[1]


def process(image, args, callback):
    from modules.sdjourney_tab import lowvram

    if not lowvram:
        gs.data["models"]["refiner"].to("cuda")

    """
    Process the chosen image. For demonstration, just display the image.
    Modify this function as needed.
    """

    seed = int(secrets.randbelow(99999999999999))
    generator = torch.Generator("cuda").manual_seed(seed)
    args["image"] = image
    args["generator"] = generator

    image = gs.data["models"]["refiner"](**args).images[0]

    callback(args["num_inference_steps"], None, None)
    st.session_state.upscaled_image = image
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    #return image


def process_image(latent, selected_values, callback):
    scheduler = selected_values['Scheduler']
    scheduler_enum = SchedulerType(scheduler)

    if scheduler_enum == SchedulerType.DPM_ANCESTRAL:
        scheduler_enum = SchedulerType.EULER_A

    get_scheduler(gs.data["models"]["refiner"], scheduler_enum)
    args = get_generation_args(selected_values, gs.data["models"]["refiner"])
    process(image=latent, args=args, callback=callback)

def load_pipeline(model_choice):

    print("LOADED:", gs.base_loaded)

    if gs.base_loaded != model_choice:
        if model_choice == "XL":
            base_pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
                use_safetensors=True
            )

            base_pipe.unet.to(memory_format=torch.channels_last)  # in-place operation
            # base_pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).to("cuda")
            from modules.sdjourney_tab import lowvram

            if lowvram:
                base_pipe.enable_model_cpu_offload()
                base_pipe.enable_vae_slicing()
                base_pipe.enable_vae_tiling()
            else:
                base_pipe.disable_attention_slicing()

            def replace_call(pipe, new_call):
                def call_with_self(*args, **kwargs):
                    return new_call(pipe, *args, **kwargs)

                return call_with_self

            base_pipe.generate = replace_call(base_pipe, new_call)
            gs.data["models"]["base"] = base_pipe
            print("XL LOADED")
        elif model_choice == "Kandinsky":
            pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder",
                                                             torch_dtype=torch.float16)
            gs.data["models"]["base"] = pipe
            print("Kandinsky loaded")
        if 'refiner' not in gs.data['models']:
            from modules.sdjourney_tab import lowvram

            refiner_pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                # text_encoder_2=gs.data["models"]["base"].text_encoder_2,
                # vae=gs.data["models"]["base"].vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            if lowvram:
                refiner_pipe.enable_model_cpu_offload()
                refiner_pipe.enable_vae_slicing()
                refiner_pipe.enable_vae_tiling()
            else:
                refiner_pipe.disable_attention_slicing()
            refiner_pipe.unet.to(memory_format=torch.channels_last)  # in-place operation
            gs.data["models"]["refiner"] = refiner_pipe
            print("Refiner Loaded")


        gs.base_loaded = model_choice

def preview_latents(latents):
    if "rgb_factor" not in gs.data:
        gs.data["rgb_factor"] = torch.tensor([
            #   R        G        B
            [0.298, 0.207, 0.208],  # L1
            [0.187, 0.286, 0.173],  # L2
            [-0.158, 0.189, 0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ], dtype=latents.dtype, device=latents.device)
    for idx, latent in enumerate(latents):
        latent_image = latent.permute(1, 2, 0) @ gs.data["rgb_factor"]

        latents_ubyte = (((latent_image + 1) / 2)
                         .clamp(0, 1)  # change scale from -1..1 to 0..1
                         .mul(0xFF)  # to 0..255
                         .byte()).cpu()
        rgb_image = latents_ubyte.numpy()[:, :, ::-1]
        image = Image.fromarray(rgb_image)
        st.session_state.image_placeholders[idx].image(image, width=image.size[0] * 4)
# from streamlit_drawable_canvas import st_canvas

def inpaint_image(image, mask):
    return image

def plugin_tab(*args, **kwargs):

    refresh = None
    col1, col2 = st.columns([2, 3])
    # Initialize the function queue
    if 'function_queue' not in st.session_state:
        st.session_state.function_queue = deque()
    # Check and initialize session states
    if 'upscaled_image' not in st.session_state:
        st.session_state.upscaled_image = None

    if 'images' not in st.session_state:
        st.session_state.images = []
    with col1:
        progress_bar = st.progress(0)
        def callback(i, t, latents):
            normalized_i = i / steps
            progress_bar.progress(normalized_i)
            if latents != None:
                preview_latents(latents)
        selected_values = dynamic_controls(controls_config)
    with col2:
        # st.session_state.image = st_canvas(
        #     fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        #     stroke_width=stroke_width,
        #     stroke_color=stroke_color,
        #     background_color=bg_color,
        #     background_image=st.session_state.upscaled_image if st.session_state.upscaled_image else None,
        #     update_streamlit=realtime_update,
        #     width=1024,
        #     height=1024,
        #     drawing_mode=drawing_mode,
        #     point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        #     key="canvas",
        # )
        # if st.session_state.image.image_data is not None:
        #     st.image(st.session_state.image.image_data)
            # if st.button("Inpaint"):
            #     st.session_state.upscaled_image = inpaint_image(st.session_state.upscaled_image, st.session_state.image.image_data )
        st.session_state.image = st.empty()

        steps = selected_values['Steps']
        generate_button = st.button("Generate", key="sd_journey_gen")
        num_images = len(st.session_state["images"])
        cols = st.columns(2)
        if 'image_placeholders' not in st.session_state:
            for j in range(len(cols)):
                st.session_state.image_placeholders = [cols[j].empty() for _ in range(4)]
        if num_images > 0:
            # Determine the number of rows and columns for the grid
            num_rows = int(np.ceil(np.sqrt(num_images)))
            for i in range(num_rows):
                  # Assuming a grid with 2 columns for simplicity
                for j in range(2):    # Loop over the 2 columns
                    index = i * 2 + j
                    if index < num_images:  # Check if we haven't exceeded the number of images
                        cols[j].image(st.session_state["images"][index])
                        if cols[j].button(f"Refine {index + 1}"):
                            process_image(st.session_state["latents"][index], selected_values, callback)
    with col1:
        load_btn = st.button("Pre-Load Models")
        # Display upscaled image if it exists
        if st.session_state.upscaled_image:
            st.image(st.session_state.upscaled_image, caption="Refined Image")

    if generate_button:
        model_choice = selected_values['BasePipeline']
        load_pipeline(model_choice)

        args = get_generation_args(selected_values, gs.data["models"]["base"])
        steps = args["num_inference_steps"]
        scheduler = selected_values['Scheduler']
        scheduler_enum = SchedulerType(scheduler)
        get_scheduler(gs.data["models"]["base"], scheduler_enum)
        generate(callback=callback, args=args)
    # while st.session_state.function_queue:
    #     func = st.session_state.function_queue.popleft()
    #     func()
    #
    #     torch.cuda.empty_cache()
    #     torch.cuda.ipc_collect()

        refresh = True
    if refresh:
        st.experimental_rerun()
        refresh = None