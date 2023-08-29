import copy
import gc
import hashlib
import json
import os

import streamlit as st
from huggingface_hub import hf_hub_download
from streamlit_extras.grid import grid
import requests
from PIL import Image
from io import BytesIO


def load_image_from_url(url):
    # Create a hash of the URL to use as the filename
    url_hash = hashlib.md5(url.encode()).hexdigest()

    # Define the local cache path
    os.makedirs('img_cache', exist_ok=True)
    cache_path = f"img_cache/{url_hash}.jpg"

    # Check if the image is in local cache
    if os.path.exists(cache_path):
        return Image.open(cache_path)

    # Download the image and save it to the cache
    response = requests.get(url)
    with open(cache_path, 'wb') as f:
        f.write(response.content)

    return Image.open(cache_path)

def display_grid(data):
    chunk_size = 4
    num_chunks = (len(data) + chunk_size - 1) // chunk_size

    # Define the grid layout
    layout = [1] * chunk_size * 2  # Two rows (image + button) for each chunk
    my_grid = grid([1,1,1,1], vertical_align="center")

    for i in range(0, len(data), chunk_size):
        # Display images for this chunk
        for j in range(chunk_size):
            # if i + j >= len(data):
            #     break

            item = data[i + j]
            if item["image"].startswith("http"):
                # The image is a URL
                img = load_image_from_url(item["image"])
            else:
                # The image is a local file
                img = Image.open(item["image"])

            my_grid.image(img, caption=item["title"])

        # Display buttons for this chunk
        for j in range(chunk_size):
            # if i + j >= len(data):
            #     break
            index = i + j
            item = data[i + j]
            if my_grid.button(f"Select {item['title']}", use_container_width=True):
                st.session_state.lora_select = (item["repo"], item["trigger_word"], item["weights"], index, item['is_compatible'])
                load_lora()

from main import singleton as gs
def load_lora():

    if "base" not in gs.data['models']:
        from modules.sdjourney_tab import load_pipeline
        load_pipeline("XL")

    if "base_copy" not in gs.data['models']:
        gs.data['models']['base_copy'] = copy.deepcopy(gs.data['models']['base'].to('cpu'))

    print(st.session_state.lora_select)
    sdxl_loras = st.session_state.sdxl_loras

    repo_name = st.session_state.lora_select[0]
    weight_name = st.session_state.lora_select[2]
    full_path_lora = gs.data['saved_names'][st.session_state.lora_select[3]]

    cross_attention_kwargs = None
    if st.session_state.last_lora != repo_name:
        if st.session_state.last_merged:
            del gs.data['models']['base']
            gc.collect()
            gs.data['models']['base'] = copy.deepcopy(gs.data['models']['base_copy'])
            gs.data['models']['base'].to('cpu')
        else:
            gs.data['models']['base'].unload_lora_weights()
        is_compatible = st.session_state.lora_select[4]

        if is_compatible:
            gs.data['models']['base'].load_lora_weights(full_path_lora)
        else:
            # merge_incompatible_lora(full_path_lora, lora_scale)
            last_merged = True

    st.session_state.last_lora = repo_name



# def merge_incompatible_lora(full_path_lora, lora_scale):
#     for weights_file in [full_path_lora]:
#         if ";" in weights_file:
#             weights_file, multiplier = weights_file.split(";")
#             multiplier = float(multiplier)
#         else:
#             multiplier = lora_scale
#
#         lora_model, weights_sd = lora.create_network_from_weights(
#             multiplier,
#             full_path_lora,
#             pipe.vae,
#             pipe.text_encoder,
#             pipe.unet,
#             for_inference=True,
#         )
#         lora_model.merge_to(
#             pipe.text_encoder, pipe.unet, weights_sd, torch.float16, "cuda"
#         )
#         del weights_sd
#         del lora_model
#         gc.collect()




plugin_info = {"name": "XL Lora Browser"}

def plugin_tab(*args, **kwargs):
    if 'last_merged' not in st.session_state:
        st.session_state.last_merged = False
    if "last_lora" not in st.session_state:
        st.session_state.last_lora = ""

    if "sdxl_loras_data" not in gs.data:
        with open("config/xl_loras.json", "r") as file:
            gs.data['sdxl_loras_data'] = json.load(file)
            gs.data['sdxl_loras'] = [
                {
                    "image": item["image"],
                    "title": item["title"],
                    "repo": item["repo"],
                    "trigger_word": item["trigger_word"],
                    "weights": item["weights"],
                    "is_compatible": item["is_compatible"],
                }
                for item in gs.data['sdxl_loras_data']
            ]
            gs.data['saved_names'] = [
                hf_hub_download(item["repo"], item["weights"]) for item in gs.data['sdxl_loras']
            ]

    display_grid(gs.data['sdxl_loras_data'])