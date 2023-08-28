# Stable Diffusion XL / Midjourney experience
from collections import deque
import streamlit as st
import numpy as np
from PIL import Image
import secrets
from enum import Enum


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

from extras import singleton as gs

plugin_info = {"name": "SDJourney"}

ON = True
OFF = False
global lowvram

lowvram = ON

def generate(args, callback):

    target_device = "cuda"
    if not lowvram:
        if gs.data["models"]["base"].device.type != target_device:
            gs.data["models"]["base"].to(target_device)
    args["callback"] = callback
    if isinstance(gs.data["models"]["base"], KandinskyV22CombinedPipeline):
        st.session_state["image_list"] = gs.data["models"]["base"](**args).images
        return st.session_state["image_list"]
    else:
        result = gs.data["models"]["base"].generate(**args)
    st.session_state.latents = result[0].clone().detach()
    callback(args["num_inference_steps"], None, None)
    # for i in range(len(result[1])):
    #     st.session_state.image_placeholders[i].image(result[1][i])


    st.session_state["image_list"] = result[1]
    return result[1]


def process(image, args, callback):
    if gs.data["models"]["refiner"].device.type != "cuda":
        gs.data["models"]["refiner"].to("cuda")

    """
    Process the chosen image. For demonstration, just display the image.
    Modify this function as needed.
    """

    seed = int(secrets.randbelow(99999999999999))
    generator = torch.Generator("cuda").manual_seed(seed)
    args["image"] = image.to('cuda')


    args["generator"] = generator

    image = gs.data["models"]["refiner"](**args).images[0]

    callback(args["num_inference_steps"], None, None)
    st.session_state.upscaled_image = image

    #return image


def process_image(latent, selected_values, callback):

    scheduler = selected_values['Scheduler']
    scheduler_enum = SchedulerType(scheduler)

    if scheduler_enum == SchedulerType.DPM_ANCESTRAL:
        scheduler_enum = SchedulerType.EULER_A

    get_scheduler(gs.data["models"]["refiner"], scheduler_enum)
    args = get_generation_args(selected_values, gs.data["models"]["refiner"])
    st.session_state.function_queue.append(lambda: process(image=latent, args=args, callback=callback))

def load_pipeline(model_choice):
    if gs.base_loaded != model_choice:
        if model_choice == "XL":
            gs.data["models"]["base"] = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
                use_safetensors=True, device_map="auto"
            )

            gs.data["models"]["base"].unet.to(memory_format=torch.channels_last)  # in-place operation
            # base_pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).to("cuda")
            # gs.data["models"]["base"].enable_vae_slicing()
            # gs.data["models"]["base"].enable_vae_tiling()
            gs.data["models"]["base"].vae.enable_slicing()
            #gs.data["models"]["base"].vae.enable_tiling()

            if lowvram:
                gs.data["models"]["base"].enable_model_cpu_offload()
            else:
                gs.data["models"]["base"].disable_attention_slicing()

            def replace_call(pipe, new_call):
                def call_with_self(*args, **kwargs):
                    return new_call(pipe, *args, **kwargs)

                return call_with_self

            gs.data["models"]["base"].generate = replace_call(gs.data["models"]["base"], new_call)
            # gs.data["models"]["base"] = base_pipe
            print("XL LOADED")
        elif model_choice == "Kandinsky":
            gs.data["models"]["base"] = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder",
                                                             torch_dtype=torch.float16).to('cuda')
            #gs.data["models"]["base"] = pipe
            print("Kandinsky loaded")
        if 'refiner' not in gs.data['models']:

            gs.data["models"]["refiner"] = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                # text_encoder_2=gs.data["models"]["base"].text_encoder_2,
                # vae=gs.data["models"]["base"].vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                device_map="auto"
            )
            # gs.data["models"]["refiner"].enable_vae_slicing()
            # gs.data["models"]["refiner"].enable_vae_tiling()
            gs.data["models"]["refiner"].vae.enable_slicing()
            gs.data["models"]["refiner"].vae.enable_tiling()
            if lowvram:
                gs.data["models"]["refiner"].enable_model_cpu_offload()
            else:
                gs.data["models"]["refiner"].disable_attention_slicing()
            # refiner_pipe.unet.to(memory_format=torch.channels_last)  # in-place operation
            # gs.data["models"]["refiner"] = refiner_pipe
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
        #st.session_state.image_placeholders[idx].image(image, width=image.size[0] * 4)
# from streamlit_drawable_canvas import st_canvas

def inpaint_image(image, mask):
    return image


def plugin_tab(tabs, tab_names):
    refresh = None
    col1_journey, col2_journey = st.columns([2, 3])

    # Initialize the function queue
    if 'function_queue' not in st.session_state:
        st.session_state.function_queue = deque()

    # Check and initialize session states
    if 'upscaled_image' not in st.session_state:
        st.session_state.upscaled_image = None

    if 'image_list' not in st.session_state:
        st.session_state.image_list = []

    with col1_journey:
        progress_bar = st.progress(0)

        def callback(i, t, latents):
            normalized_i = i / steps
            progress_bar.progress(normalized_i)
            if latents != None:
                preview_latents(latents)

        selected_values = dynamic_controls(controls_config)

    with col2_journey:
        st.session_state.journey_image = st.empty()
        steps = selected_values['Steps']
        generate_button = st.button("Generate", key="sd_journey_gen")
        num_images = len(st.session_state["image_list"])

        # Assuming a 2x2 grid for simplicity
        num_rows = 2
        num_cols = 2

        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col in range(num_cols):
                index = row * num_cols + col
                if index < num_images:
                    cols[col].image(st.session_state["image_list"][index])
                    if cols[col].button(f"Refine {index + 1}"):
                        latent = st.session_state["latents"][index].clone().cuda()
                        process_image(latent, selected_values, callback)
    with col1_journey:
        load_btn = st.button("Pre-Load Models")

        # Display upscaled image if it exists
        if st.session_state.upscaled_image:
            st.image(st.session_state.upscaled_image, caption="Refined Image")

    if load_btn:
        model_choice = selected_values['BasePipeline']
        load_pipeline(model_choice)

    if generate_button:
        args = get_generation_args(selected_values, gs.data["models"]["base"])
        steps = args["num_inference_steps"]
        scheduler = selected_values['Scheduler']
        scheduler_enum = SchedulerType(scheduler)
        get_scheduler(gs.data["models"]["base"], scheduler_enum)
        st.session_state.function_queue.append(lambda: generate(callback=callback, args=args))

    while st.session_state.function_queue:
        func = st.session_state.function_queue.popleft()
        func()
        refresh = True
        st.experimental_rerun()
