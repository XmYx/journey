import os
from typing import List

import streamlit as st
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, KandinskyV22CombinedPipeline

from extras import resize_right, interp_methods
from extras.sdjourney_backend import aspect_ratios, SchedulerType, get_scheduler, scheduler_type_values
from extras.styles import apply_style, style_keys
from main import singleton as gs


def list_model_files(directory="models/loras"):
    """
    Lists all files in the specified directory with the extensions: ckpt, bin, pt, safetensors.

    Parameters:
    - directory: The directory path to search for files.

    Returns:
    - A list of file paths with the specified extensions.
    """

    # List of valid extensions
    valid_extensions = ['.ckpt', '.bin', '.pt', '.safetensors']

    # Get all files in the directory with the specified extensions
    file_list = [os.path.join(directory, f) for f in os.listdir(directory)
                 if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1] in valid_extensions]

    return file_list

def load_lora(args):
    try:
        path = f"models/loras/{args['select_lora']}"
        gs.data["models"]["base"].load_lora_weights(args['select_lora'])
    except Exception as e:
        print("Could not load LORA", repr(e))
    return args

st_widgets = {
    "text":"text_input",
    "text_multi":"text_area",
    "number":"number",
    "slider":"slider",
    "dropdown":"selectbox",
    "checkbox":"checkbox",
    "radio":"multiselect",
}

class BaseWidget:

    widget_type: str = "text" # ["text", "number", "slider", "dropdown", "checkbox"]
    multiline: bool = False
    precision: str = "int"
    def __init__(self):
        if self.widget_type == "text":
            self.widget_type = self.widget_type if not self.multiline else "text_multi"
        self.widget = getattr(st, st_widgets[self.widget_type])
    def get(self):
        return self.widget

    def get_value(self):
        if hasattr(self.widget, 'value'):
            return self.widget.value()
class BaseBlock:
    widgets = List[BaseWidget]
class BlockHolder:
    blocks: List[BaseBlock] = None


blocks = {
    "Loader": {
        "category": "starter",
        "controls": {
            "select_model": {
                "type": "selectbox",
                "expose": True,
                "params": {"options":["XL", "Kandinsky"]}
            },
        },
        "fn": "load_xl"
    },
    "Lora Loader": {
        "category": "middle",
        "controls": {
            "select_lora": {
                "type": "selectbox",
                "expose": True,
                "params": {"options":list_model_files()}
            },
        },
        "fn": "load_lora"
    },

    "Prompt": {
        "category": "middle",
        "controls": {
            "prompt": {
                "type": "text_area",
                "expose": True,
                "params": {"value": ""}
            },
        },
        "fn": "dummy_fn"
    },
    "Classic Prompt": {
        "category": "middle",
        "controls": {
            "prompt_2": {
                "type": "text_area",
                "expose": True,
                "params": {"value": ""}
            },
        },
        "fn": "dummy_fn"
    },
    "Negative Prompt": {
        "category": "middle",
        "controls": {
            "negative_prompt": {
                "type": "text_area",
                "expose": True,
                "params": {"value": ""}
            },
        },
        "fn": "dummy_fn"
    },
    "Classic Negative Prompt": {
        "category": "middle",
        "controls": {
            "negative_prompt_2": {
                "type": "text_area",
                "expose": True,
                "params": {"value": ""}
            },
        },
        "fn": "dummy_fn"
    },
    "Resize": {
        "category": "middle",
        "controls": {
            "width": {
                "type": "slider",
                "expose": False,
                "params": {"value": 1024,
                           "min_value":8,
                           "max_value":4096,
                           "step":8}
            },
            "height": {
                "type": "slider",
                "expose": False,
                "params": {"value": 1024,
                           "min_value":8,
                           "max_value":4096,
                           "step":8}
            },
        },
        "fn": "resize"
    },


    "Params": {
        "category": "middle",
        "controls": {
            "resolution": {
                "type": "selectbox",
                "expose": False,
                "params": {"options": list(aspect_ratios.keys())}
            },
            "num_inference_steps": {
                "type": "slider",
                "expose": False,
                "params": {"value": 25}
            },
            "guidance_scale": {
                "type": "slider",
                "expose": False,
                "params": {"value": 5.00,
                           "min_value":0.01,
                           "max_value":25.00}
            },
            "mid_point": {
                "type": "slider",
                "expose": False,
                "params": {"value": 0.80,
                           "min_value":0.01,
                           "max_value":1.00}
            },
            "num_images_per_prompt": {
                "type": "slider",
                "expose": False,
                "params": {"value": 1,
                           "min_value":1,
                           "max_value":125}
            },
        },
        "fn": "dummy_fn"
    },

    "Generate": {
        "category": "middle",
        "fn": "generate",
        "controls": {
            "show_image": {
                "type": "checkbox",
                "expose": True,
                "params": {"value": True}
            },
            "scheduler": {
                "type": "selectbox",
                "expose": True,
                "params": {"options": scheduler_type_values}
            },
        },
    },
    "Refine": {
        "category": "middle",
        "fn": "refine",
        "controls": {
            "show_image": {
                "type": "checkbox",
                "expose": True,
                "params": {"value": True}
            },
            "scheduler": {
                "type": "selectbox",
                "expose": True,
                "params": {"options": scheduler_type_values}
            },

        },
    },
    "XL Style": {
        "category": "middle",
        "controls": {
            "style": {
                "type": "selectbox",
                "expose": True,
                "params": {"options": style_keys}
            },

        },
        "fn": "style"
    },

    "Add Noise": {
        "category": "middle",
        "controls": {
            "noise_type": {
                "type": "selectbox",
                "expose": True,
                "params": {"options": ["gaussian", "salt_and_pepper"]}
            },
            "multiplier": {
                "type": "slider",
                "expose": False,
                "params": {"value": 0.80,
                           "min_value": 0.01,
                           "max_value": 1.00}
            },
        },
        "fn": "add_noise"
    },
}

def style(args):
    prompt = args.get('prompt', "")
    negative_prompt = args.get('negative_prompt', "")
    style = args.get('style', 'None')
    args['prompt'], args['negative_prompt'] = apply_style(style, prompt, negative_prompt)
    return args
def resize(args):

    w = args["width"]
    h = args["height"]
    latents = gs.data.get("latents", None)
    result = []
    if latents is not None:
        for latent in latents:
            print(latent.shape)
            new = resize_right.resize(latent, scale_factors=None, out_shape=[latent.shape[0],h // 8, w // 8],
                    interp_method=interp_methods.cubic, support_sz=None,
                    antialiasing=True, by_convs=False, scale_tolerance=None,
                    max_numerator=10, pad_mode='constant')
            result.append(new)
    gs.data["latents"] = result
    return args
def load_xl(args):

    from modules.sdjourney import load_pipeline
    load_pipeline(args['select_model'])

    return args

def check_args(args, pipe):
    gen_args = {
        "prompt": args.get("prompt", "test"),
        "num_inference_steps": args.get("num_inference_steps", 10)
    }

    if isinstance(pipe, (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline)):
        gen_args["prompt_2"] = args.get('prompt_2')
        gen_args["negative_prompt_2"] = args.get('negative_prompt_2')

    if isinstance(pipe, StableDiffusionXLPipeline):
        gen_args["width"], gen_args["height"] = aspect_ratios.get(args.get('resolution', "1024 x 1024 (1:1 Square)"))
        gen_args["denoising_end"] = args.get('mid_point', 0.87)
        gen_args["num_images_per_prompt"] = int(args.get('num_images_per_prompt', 1))

    elif isinstance(pipe, StableDiffusionXLImg2ImgPipeline):
        gen_args["strength"] = args.get('strength', 0.3)
        gen_args["denoising_start"] = args.get('mid_point', 0.87)
        #gen_args["image"] = gs.data.get('latents')

    elif isinstance(pipe, KandinskyV22CombinedPipeline):
        gen_args["width"], gen_args["height"] = aspect_ratios.get(args.get('resolution', "1024 x 1024 (1:1 Square)"))
        gen_args["num_images_per_prompt"] = int(args.get('num_images_per_prompt', 1))

    scheduler = args['scheduler']
    scheduler_enum = SchedulerType(scheduler)
    pipe = get_scheduler(pipe, scheduler_enum)


    return gen_args, pipe
def generate(args):
    target_device = "cuda"
    if gs.data["models"]["base"].device.type != target_device:
        gs.data["models"]["base"].to(target_device)

    gen_args, pipe = check_args(args, gs.data["models"]["base"])
    progressbar = args.get('progressbar', None)

    def callback(i, t, latents):
        if progressbar:
            normalized_i = i / args.get('num_inference_steps', 10)
            progressbar.progress(normalized_i)

    gen_args["callback"] = callback

    if args["show_image"]:
        if isinstance(pipe, StableDiffusionXLPipeline):
            result = pipe.generate(**gen_args)
            gs.data["latents"] = result[0]
            result_dict = {"result_image": result[1]}
            st.session_state.preview = result[1][0]

        else:
            result_dict = {"result_image": pipe(**gen_args).images}
            gs.data["latents"] = result_dict['result_image']
            st.session_state.preview = result_dict['result_image'][0]
    else:
        gen_args['output_type'] = 'latent'
        result = pipe(**gen_args).images
        gs.data["latents"] = result
        result_dict = {}
    result_dict = {**args, **result_dict}
    if progressbar:
        progressbar.progress(1.0)

    return result_dict

def refine(args):
    target_device = "cuda"
    # if gs.data["models"]["refiner"].device.type != target_device:
    #     gs.data["models"]["refiner"].to(target_device)
    gs.data["models"]["refiner"].to("cuda")
    gen_args, pipe = check_args(args, gs.data["models"]["refiner"])
    #gen_args['image'] = gen_args['image'].to('cpu')
    latents = gs.data.get('latents')
    images = []
    for latent in latents:
        gen_args['image'] = latent
        result = pipe(**gen_args)
        images.append(result.images[0])

    if args["show_image"]:
        result_dict = {"result_image":images}
    else:
        result_dict = {"hidden_result_image":images}

    result_dict = {**args, **result_dict}
    return result_dict


def dummy_fn(args):

    #print("starter module data", starter_text)

    return args

def middle_fn(middle_text):
    print(middle_text)
    return middle_text

def end_fn(end_text):
    return end_text


def add_noise(args):
    """
    Adds noise to an image.

    Parameters:
    - img: A PIL Image or a PyTorch tensor.
    - multiplier: A value indicating the strength of the noise.
    - noise_type: The type of noise to add ("gaussian" or "salt_and_pepper").

    Returns:
    - A PIL Image or tensor with noise added.
    """
    imgs = args['result_image']
    multiplier = args['multiplier']
    noise_type = args['noise_type']
    results = []
    convert = False
    # Convert PIL Image to tensor
    for img in imgs:
        if isinstance(img, Image.Image):
            img = torch.tensor(np.array(img).transpose(2, 0, 1)).float() / 255.0
            convert = True
        # Ensure the tensor is on the CPU
        img = img.cpu()

        if noise_type == "gaussian":
            noise = torch.randn_like(img) * multiplier
        elif noise_type == "salt_and_pepper":
            noise = torch.where(torch.rand_like(img), torch.tensor(-1.0), torch.tensor(1.0)) * multiplier
        else:
            raise ValueError("Invalid noise_type. Choose from ['gaussian', 'salt_and_pepper']")

        # Add noise and clip values between 0 and 1
        noisy_img = torch.clamp(img + noise, 0, 1)

        # Convert back to PIL Image format if needed
        if convert:
            noisy_img = Image.fromarray((noisy_img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        results.append(noisy_img)
    return {"result_image":results}