import secrets
from enum import Enum

import streamlit as st
import torch
from PIL import Image
import numpy as np

from diffusers import ( StableDiffusionInpaintPipeline, DDIMScheduler, HeunDiscreteScheduler, KDPM2DiscreteScheduler,
                        KDPM2AncestralDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler, EulerDiscreteScheduler,
                        EulerAncestralDiscreteScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler)

pipe = None
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.enabled = False

class SchedulerType(Enum):
    DDIM = "ddim"
    HEUN = "heun"
    DPM_DISCRETE = "dpm_discrete"
    DPM_ANCESTRAL = "dpm_ancestral"
    LMS = "lms"
    PNDM = "pndm"
    EULER = "euler"
    EULER_A = "euler_a"
    DPMPP_SDE_ANCESTRAL = "dpmpp_sde_ancestral"
    DPMPP_2M = "dpmpp_2m"

scheduler_type_values = [item.value for item in SchedulerType]

def do_inpaint(prompt, init_image, mask_image, scheduler_type, steps, guidance_scale, strength, seed, selected_repo, model_name="inpaint"):

    scheduler = SchedulerType(scheduler_type)

    try:
        seed = int(seed)
    except:
        seed = secrets.randbelow(999999999)
    if seed == 0:
        seed = secrets.randbelow(999999999)
    generator = torch.Generator("cuda").manual_seed(seed)

    if "pipe" not in st.session_state or st.session_state["model_name"] != model_name:
    #if pipe == None:

        print(selected_repo)

        #st.session_state["pipe"] = StableDiffusionInpaintPipeline.from_pretrained(selected_repo, torch_dtype=torch.float16).to("cuda")
        st.session_state["pipe"] = StableDiffusionInpaintPipeline.from_single_file("C:/Users/mix/Downloads/realisticVisionV51_v51VAE-inpainting.safetensors",
                                                                                   torch_dtype=torch.float16,
                                                                                   load_safety_checker=False).to("cuda")
        st.session_state["model_name"] = model_name
    with torch.inference_mode():
        get_scheduler(st.session_state["pipe"], scheduler)
        image = st.session_state["pipe"](prompt=prompt,
                     image=init_image,
                     mask_image=mask_image,
                     width=init_image.size[0],
                     height=init_image.size[1],
                     strength=strength,
                     guidance_scale=guidance_scale,
                     num_inference_steps=steps,
                     generator=generator).images[0]
    return image

def scale_and_paste(target_img, source_img, scale, offset_x, offset_y, expand=8):
    # Resize the source image
    source_img = source_img.resize((int(source_img.width * scale), int(source_img.height * scale)))

    # Create a new image of the target size
    new_img = Image.new('RGB', (target_img.width, target_img.height))

    # Paste the scaled source image onto the new image
    new_img.paste(source_img, (offset_x, offset_y))

    # Create a mask of the empty area
    mask = np.full((new_img.height, new_img.width), 255)
    mask[offset_y+expand:(source_img.height-expand+offset_y), offset_x+expand:(source_img.width-expand+offset_x)] = 0
    mask = Image.fromarray(mask).convert("RGB")

    print(new_img.size, mask.size)

    return new_img, mask

def get_scheduler(pipe, scheduler: SchedulerType):
    scheduler_mapping = {
        SchedulerType.DDIM: DDIMScheduler.from_config,
        SchedulerType.HEUN: HeunDiscreteScheduler.from_config,
        SchedulerType.DPM_DISCRETE: KDPM2DiscreteScheduler.from_config,
        SchedulerType.DPM_ANCESTRAL: KDPM2AncestralDiscreteScheduler.from_config,
        SchedulerType.LMS: LMSDiscreteScheduler.from_config,
        SchedulerType.PNDM: PNDMScheduler.from_config,
        SchedulerType.EULER: EulerDiscreteScheduler.from_config,
        SchedulerType.EULER_A: EulerAncestralDiscreteScheduler.from_config,
        SchedulerType.DPMPP_SDE_ANCESTRAL: DPMSolverSinglestepScheduler.from_config,
        SchedulerType.DPMPP_2M: DPMSolverMultistepScheduler.from_config
    }

    new_scheduler = scheduler_mapping[scheduler](pipe.scheduler.config)
    pipe.scheduler = new_scheduler

    return pipe


diffusers_models = [
    {"name": "stable-diffusion-v1-5", "repo": "runwayml/stable-diffusion-v1-5"},
    {"name": "stable-diffusion-v2-inpaint", "repo": "stabilityai/stable-diffusion-2-inpainting"},
    {"name": "revAnimated", "repo": "danbrown/RevAnimated-v1-2-2"},
    {"name": "Realistic_Vision_V1.4", "repo": "SG161222/Realistic_Vision_V1.4"},
    {"name": "stable-diffusion-v1-4", "repo": "CompVis/stable-diffusion-v1-4"},
    {"name": "openjourney", "repo": "prompthero/openjourney"},
    {"name": "stable-diffusion-2-1-base", "repo": "stabilityai/stable-diffusion-2-1-base"},
    {"name": "stable-diffusion-inpainting", "repo": "runwayml/stable-diffusion-inpainting"},
    {"name": "waifu-diffusion", "repo": "hakurei/waifu-diffusion"},
    {"name": "stable-diffusion-2-1", "repo": "stabilityai/stable-diffusion-2-1"},
    {"name": "dreamlike-photoreal-2.0", "repo": "dreamlike-art/dreamlike-photoreal-2.0"},
    {"name": "anything-v3.0", "repo": "Linaqruf/anything-v3.0"},
    {"name": "DreamShaper", "repo": "Lykon/DreamShaper"},
    {"name": "dreamlike-diffusion-1.0", "repo": "dreamlike-art/dreamlike-diffusion-1.0"},
    {"name": "stable-diffusion-2", "repo": "stabilityai/stable-diffusion-2"},
    {"name": "vox2", "repo": "plasmo/vox2"},
    {"name": "openjourney-v4", "repo": "prompthero/openjourney-v4"},
    {"name": "sd-pokemon-diffusers", "repo": "lambdalabs/sd-pokemon-diffusers"},
    {"name": "Protogen_x3.4_Official_Release", "repo": "darkstorm2150/Protogen_x3.4_Official_Release"},
    {"name": "Taiyi-Stable-Diffusion-1B-Chinese-v0.1", "repo": "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"},
    {"name": "dreamlike-anime-1.0", "repo": "dreamlike-art/dreamlike-anime-1.0"},
    {"name": "Analog-Diffusion", "repo": "wavymulder/Analog-Diffusion"},
    {"name": "stable-diffusion-2-base", "repo": "stabilityai/stable-diffusion-2-base"},
    {"name": "trinart_stable_diffusion_v2", "repo": "naclbit/trinart_stable_diffusion_v2"},
    {"name": "vintedois-diffusion-v0-1", "repo": "22h/vintedois-diffusion-v0-1"},
    {"name": "stable-diffusion-v1-2", "repo": "CompVis/stable-diffusion-v1-2"},
    {"name": "Arcane-Diffusion", "repo": "nitrosocke/Arcane-Diffusion"},
    {"name": "SomethingV2_2", "repo": "NoCrypt/SomethingV2_2"},
    {"name": "EimisAnimeDiffusion_1.0v", "repo": "eimiss/EimisAnimeDiffusion_1.0v"},
    {"name": "Protogen_x5.8_Official_Release", "repo": "darkstorm2150/Protogen_x5.8_Official_Release"},
    {"name": "Nitro-Diffusion", "repo": "nitrosocke/Nitro-Diffusion"},
    {"name": "anything-midjourney-v-4-1", "repo": "Joeythemonster/anything-midjourney-v-4-1"},
    {"name": "anything-v5", "repo": "stablediffusionapi/anything-v5"},
    {"name": "portraitplus", "repo": "wavymulder/portraitplus"},
    {"name": "epic-diffusion", "repo": "johnslegers/epic-diffusion"},
    {"name": "noggles-v21-6400-best", "repo": "alxdfy/noggles-v21-6400-best"},
    {"name": "Future-Diffusion", "repo": "nitrosocke/Future-Diffusion"},
    {"name": "photorealistic-fuen-v1", "repo": "claudfuen/photorealistic-fuen-v1"},
    {"name": "Comic-Diffusion", "repo": "ogkalu/Comic-Diffusion"},
    {"name": "Ghibli-Diffusion", "repo": "nitrosocke/Ghibli-Diffusion"},
    {"name": "OrangeMixs", "repo": "WarriorMama777/OrangeMixs"},
    {"name": "children_stories_inpainting", "repo": "ducnapa/children_stories_inpainting"},
    {"name": "DucHaitenAIart", "repo": "DucHaiten/DucHaitenAIart"},
    {"name": "Dungeons-and-Diffusion", "repo": "0xJustin/Dungeons-and-Diffusion"},
    {"name": "redshift-diffusion-768", "repo": "nitrosocke/redshift-diffusion-768"},
    {"name": "Anything-V3-X", "repo": "iZELX1/Anything-V3-X"},
    {"name": "anime-kawai-diffusion", "repo": "Ojimi/anime-kawai-diffusion"},
    {"name": "midjourney-v4-diffusion", "repo": "flax/midjourney-v4-diffusion"},
    {"name": "seek.art_MEGA", "repo": "coreco/seek.art_MEGA"},
    {"name": "karlo-v1-alpha", "repo": "kakaobrain/karlo-v1-alpha"},
    {"name": "edge-of-realism", "repo": "stablediffusionapi/edge-of-realism"},
    {"name": "anything-v3-1", "repo": "cag/anything-v3-1"},
    {"name": "classic-anim-diffusion", "repo": "nitrosocke/classic-anim-diffusion"},
]
plugin_info = {"name": "Outpaint"}

def plugin_tab(tabs, tab_names):
    name_to_repo = {model['name']: model['repo'] for model in diffusers_models}

    if "model_name" not in st.session_state:
        st.session_state["model_name"] = ""

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)

        col1, col2, col3 = st.columns([1,5,5])

        with col1:
            global model_name
            model_name = st.selectbox("Model", options=list(name_to_repo.keys()))
            selected_repo = name_to_repo[model_name]
            tab_name = "outpaint"
            prompt = st.text_input('Prompt', key=f"{tab_name}_prompt")
            # Allow the user to input the target size and scale
            target_width = st.number_input('Target width', min_value=64, value=768, step=8)
            target_height = st.number_input('Target height', min_value=64, value=768, step=8)
            scale = st.slider('Scale', min_value=0.1, max_value=2.0, step=0.01, value=0.4)
            offset_x = st.slider('X Offset', min_value=0, max_value=1024, step=8, value=64)
            offset_y = st.slider('Y Offset', min_value=0, max_value=1024, step=8, value=64)
            scheduler_type = st.selectbox("Scheduler", scheduler_type_values, key=f"{tab_name}_scheduler")
            steps = st.number_input("Steps", min_value=1, max_value=1000, value=50, key=f"{tab_name}_steps")
            guidance_scale = st.number_input("Guidance Scale", min_value=0.0, max_value=25.0, value=6.5, key=f"{tab_name}_guidance_scale")
            strength = st.number_input("Strength", min_value=0.0, max_value=1.0, value=1.0, key=f"{tab_name}_str")
            seed = st.text_input("Seed", value="0", key=f"{tab_name}_seed")
            expand = st.number_input("Mask Expansion", min_value=0, max_value=256, step=1, value=8)

            # Process the image
            target_image = Image.new('RGB', (target_width, target_height))
            new_image, mask = scale_and_paste(target_image, input_image, scale, offset_x, offset_y)

        with col2:
            # Display the new image and mask
            st.image(new_image, caption='New Image')
            st.image(mask, caption='Mask')
            button = st.button("Outpaint")
        with col3:
            if button:
                mask = mask.resize(new_image.size, resample=Image.Resampling.LANCZOS)
                image = st.image(do_inpaint(prompt, new_image, mask, scheduler_type, steps, guidance_scale, strength, seed, selected_repo))
