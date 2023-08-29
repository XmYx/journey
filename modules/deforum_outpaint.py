import secrets
from datetime import datetime

import streamlit as st
import torch

from deforum.avfunctions.hybridvideo.hybrid_video import image_transform_optical_flow, get_flow_from_images
from deforum.avfunctions.interpolation.RAFT import RAFT
from main import singleton as gs
from extras.block_helpers import list_model_files

plugin_info = {"name": "Deforum Outpaint"}


from PIL import Image, ImageDraw
import numpy as np

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline


# Mock functions
def load():
    if 'base' not in gs.data['models']:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True, device_map="auto"
        )



        inpaintpipe = StableDiffusionXLInpaintPipeline(vae=pipe.vae,
                                         text_encoder=pipe.text_encoder,
                                         text_encoder_2=pipe.text_encoder_2,
                                         tokenizer=pipe.tokenizer,
                                         tokenizer_2=pipe.tokenizer_2,
                                         unet=pipe.unet,
                                         scheduler=pipe.scheduler).to('cuda')
        img2imgpipe = StableDiffusionXLImg2ImgPipeline(vae=pipe.vae,
                                         text_encoder=pipe.text_encoder,
                                         text_encoder_2=pipe.text_encoder_2,
                                         tokenizer=pipe.tokenizer,
                                         tokenizer_2=pipe.tokenizer_2,
                                         unet=pipe.unet,
                                         scheduler=pipe.scheduler).to('cuda')

        gs.data["models"]["base"] = pipe
        gs.data["models"]["inpaint"] = inpaintpipe
        gs.data["models"]["img2img"] = img2imgpipe

def get_generator(seed):
    return torch.Generator('cuda').manual_seed(seed)

def generate(args):
    """Mock function to generate a random 1024x1024 image."""
    seed = args.get("seed", secrets.randbelow(9999999999999))
    prompt = args.get("prompt", "deforum animation")

    if seed == 0:
        seed = secrets.randbelow(9999999999999)

    generator = get_generator(seed)
    image = gs.data["models"]["base"](prompt=prompt, generator=generator).images[0]
    return image

def outpaint(args):
    """Mock function that just returns the input image."""
    image = args.get("image")
    mask = args.get("mask")
    prompt = args.get("prompt", "deforum animation")
    seed = args.get("seed", secrets.randbelow(9999999999999))

    if seed == 0:
        seed = secrets.randbelow(9999999999999)


    generator = get_generator(seed)

    image = gs.data["models"]["inpaint"](image=image,
                                         mask_image=mask,
                                         prompt=prompt,
                                         generator=generator).images[0]
    return image
def img2img(args):
    """Mock function that just returns the input image."""
    image = args.get("image")
    prompt = args.get("prompt", "deforum animation")
    seed = args.get("seed", secrets.randbelow(9999999999999))
    strength = 0.65
    if seed == 0:
        seed = secrets.randbelow(9999999999999)


    generator = get_generator(seed)

    image = gs.data["models"]["img2img"](image=image,
                                         strength=strength,
                                         prompt=prompt,
                                         generator=generator).images[0]
    return image

# Main logic






def create_sequence(args):

    raft_model = RAFT()
    method = 'RAFT'

    max_frames = args.get("frame_count", 5)
    preview = args.get("preview", None)

    # Initialize with a generated image
    current_image = generate(args)
    frames = []

    # Append the top half of the first generated image to frames
    top_half = current_image.crop((0, 0, 1024, 512))
    bottom_half = current_image.crop((0, 512, 1024, 1024))


    top_half_np = np.array(top_half).astype(np.uint8)
    bottom_half_np = np.array(bottom_half).astype(np.uint8)
    flow = get_flow_from_images(top_half, bottom_half, method, raft_model, prev_flow=None)
    top_half_np = image_transform_optical_flow(top_half_np, flow, 1)
    top_half = Image.fromarray(top_half_np)

    frames.append(top_half)

    for _ in range(max_frames):
        # Save the bottom half of the current image

        if preview is not None:
            preview.image(current_image)

        top_half = current_image.crop((0, 0, 1024, 512))
        bottom_half = current_image.crop((0, 512, 1024, 1024))

        top_half_np = np.array(top_half).astype(np.uint8)
        bottom_half_np = np.array(bottom_half).astype(np.uint8)
        flow = get_flow_from_images(top_half, bottom_half, method, raft_model, prev_flow=flow)
        bottom_half_np = image_transform_optical_flow(bottom_half_np, flow, 1)
        bottom_half = Image.fromarray(bottom_half_np)

        frames.append(bottom_half)

        # Use the bottom half as the top half for the next image
        new_top_half = bottom_half.copy()
        new_bottom_half = Image.new("RGB", (1024, 512), (255, 255, 255))
        next_image = Image.new("RGB", (1024, 1024))
        next_image.paste(new_top_half, (0, 0))
        next_image.paste(new_bottom_half, (0, 512))

        # Create a mask for the bottom half
        mask = Image.new("L", (1024, 1024), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([(0, 512), (1024, 1024)], fill=255)
        args["image"] = current_image
        args["mask"] = mask
        # Use the outpaint function

        use_img2img = args.get('use_img2img', False)

        if use_img2img:
            current_image = img2img(args)
        else:
            current_image = outpaint(args)

    return frames


def plugin_tab(*args, **kwargs):
    frame_count = st.number_input('Frames', min_value=1, max_value=500, value=5)
    generate_button = st.button('Generate')
    lora = st.selectbox('Lora', list_model_files())
    seed = st.number_input('Seed', value=0)
    prompt = st.text_area('Prompt')
    gif_duration = st.number_input('ms / frame', value=120, min_value=40, max_value=5000)
    method = st.toggle('Inpaint/Img2Img')
    preview = st.empty()
    if "lora_loaded" not in gs.data:
        gs.data["lora_loaded"] = ""
    if generate_button:

        load()

        if gs.data["lora_loaded"] != lora:
            gs.data["models"]["base"].load_lora_weights(lora)
            gs.data["lora_loaded"] = lora

        args = {"prompt":prompt,
                "frame_count":frame_count,
                "preview":preview,
                "seed":seed,
                'use_img2img':method}

        frames = create_sequence(args)
        st.image(frames)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = f"output/{timestamp}_sequence.gif"
        frames[0].save(gif_path, append_images=frames[1:], save_all=True, duration=gif_duration, loop=0)


