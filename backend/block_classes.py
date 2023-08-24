import torch
from PIL import Image

from backend.block_base import register_class, BaseBlock
from extras.block_helpers import check_args, style
from extras.sdjourney_backend import scheduler_type_values, aspect_ratios
from extras.styles import style_keys
from main import singleton as gs
import streamlit as st

plugin_info = {"name": "Blocks v2"}

@register_class
class SampleBlock(BaseBlock):
    def __init__(self):
        super().__init__()
        self.text('Prompt')
        self.number('Steps')

    def fn(self, data: dict) -> dict:
        prompt = self.widgets[0].value
        steps = self.widgets[1].value
        processed = f"{prompt} - {steps}"

        # Update the data dictionary
        data["processed"] = processed
        data["prompt"] = prompt
        data["steps"] = steps
        print("SAMPLE FUNCTION", data)
        return data

@register_class
class DiffusersXLLoaderBlock(BaseBlock):

    name = "SD XL Loader"

    def __init__(self):
        super().__init__()
        self.dropdown('model_select', ["XL", "Kandinsky"])
    def fn(self, data: dict) -> dict:
        from modules.sdjourney import load_pipeline

        selection = self.widgets[0].selected_index
        selection = self.widgets[0].options[selection]
        load_pipeline(selection)

        return data
@register_class
class DiffusersParamsBlock(BaseBlock):

    name = "SD Parameters"


    def __init__(self):
        super().__init__()
        self.number('Steps', 25, 1, 2, 250)
        self.number('Guidance Scale', 7.5, 0.01, 0.1, 25.0)
        self.number('Mid Point', 0.80, 0.01, 0.1, 1.0)
        self.dropdown('Aspect Ratio', list(aspect_ratios.keys()))

    def fn(self, data: dict) -> dict:
        data["num_inference_steps"] = self.widgets[0].value
        data["guidance_scale"] = self.widgets[1].value
        data["mid_point"] = self.widgets[2].value
        dropdown = self.widgets[3]
        data["resolution"] = dropdown.options[dropdown.selected_index]
        return data
@register_class
class DiffusersPromptBlock(BaseBlock):

    name = "SD Prompt"


    def __init__(self):
        super().__init__()
        self.text('prompt', multiline=True)

    def fn(self, data: dict) -> dict:
        data["prompt"] = self.widgets[0].value
        print(data)

        return data
@register_class
class DiffusersPromptStyleBlock(BaseBlock):

    name = "SD Prompt Style"


    def __init__(self):
        super().__init__()
        self.dropdown('prompt', style_keys)

    def fn(self, data: dict) -> dict:
        data["style"] = self.widgets[0].options[self.widgets[0].selected_index]
        data = style(data)
        return data
@register_class
class DiffusersSamplerBlock(BaseBlock):

    name = "SD Sampler"


    def __init__(self):
        super().__init__()
        self.dropdown('Scheduler', scheduler_type_values)
        self.checkbox('Force full sample')
    def fn(self, data: dict) -> dict:
        if hasattr(self, 'index'):
            print("Block Index", self.index)
            print("Block Amount", len(gs.data['added_blocks']))
        target_device = "cuda"
        if gs.data["models"]["base"].device.type != target_device:
            gs.data["models"]["base"].to(target_device)
        widget = self.widgets[0]
        data['scheduler'] = widget.options[widget.selected_index]
        args, pipe = check_args(data, gs.data['models']['base'])
        progressbar = False
        def callback(i, t, latents):
            if progressbar:
                normalized_i = i / args.get('num_inference_steps', 10)
                progressbar.progress(normalized_i)
            preview_latents(latents)
        args["callback"] = callback
        show_image = self.widgets[1].value
        # if hasattr(self, 'index'):
        #     show_image = self.index + 1 == len(gs.data['added_blocks'])

        print("Show Image", show_image)

        if show_image:
            result = pipe.generate(**args)
            gs.data["latents"] = result[0]
            data["result_image"] = result[1]
            st.session_state.preview = result[1][0]
            if "images" not in st.session_state:
                st.session_state.images = []
            st.session_state.images.append(result[1])
            if len(st.session_state['images']) > 8:
                if len(st.session_state['start_index']) < 8:
                    st.session_state.start_index = 8
                else:
                    st.session_state.start_index += 1

        else:
            args['output_type'] = 'latent'
            result = pipe(**args).images
            gs.data["latents"] = result
        return data


def preview_latents(latents):
    if "rgb_factor" not in gs.data:
        gs.data["rgb_factor"] = torch.tensor([
            #   R        G        B
            [0.298, 0.207, 0.208],  # L1
            [0.187, 0.286, 0.173],  # L2
            [-0.158, 0.189, 0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ], dtype=latents.dtype, device=latents.device)
    # for idx, latent in enumerate(latents):
    latent_image = latents[0].permute(1, 2, 0) @ gs.data["rgb_factor"]

    latents_ubyte = (((latent_image + 1) / 2)
                     .clamp(0, 1)  # change scale from -1..1 to 0..1
                     .mul(0xFF)  # to 0..255
                     .byte()).cpu()
    rgb_image = latents_ubyte.numpy()[:, :, ::-1]
    image = Image.fromarray(rgb_image)
    st.session_state.preview_holder.image(image, width=image.size[0] * 8)

@register_class
class DiffusersRefinerBlock(BaseBlock):

    name = "SD XL Refiner"


    def __init__(self):
        super().__init__()
        self.dropdown('Scheduler', scheduler_type_values)
    def fn(self, data: dict) -> dict:

        target_device = "cuda"
        if gs.data["models"]["refiner"].device.type != target_device:
            gs.data["models"]["refiner"].to(target_device)
        widget = self.widgets[0]
        data['scheduler'] = widget.options[widget.selected_index]
        args, pipe = check_args(data, gs.data['models']['refiner'])
        progressbar = False
        def callback(i, t, latents):
            if progressbar:
                normalized_i = i / args.get('num_inference_steps', 10)
                progressbar.progress(normalized_i)
            preview_latents(latents)
        args["callback"] = callback

        latents = gs.data.get('latents')
        images = []
        for latent in latents:
            args['image'] = latent
            result = pipe(**args)
            images.append(result.images[0])


        #gs.data["latents"] = result[0]
        data["result_image"] = images

        st.session_state.preview = images[0]

        if "images" not in st.session_state:
            st.session_state.images = []

        st.session_state.images.append(images)
        if len(st.session_state['images']) > 8:
            if len(st.session_state['start_index']) < 8:
                st.session_state.start_index = 8
            else:
                st.session_state.start_index += 1
        return data
@register_class
class CodeformersBlock(BaseBlock):
    name = "Codeformers"
    def __init__(self):
        super().__init__()
        from backend.codeformers import codeformersinference, init_codeformers

        init_codeformers()

        self.checkbox('Align Faces', True)
        self.checkbox('Enhance Background', True)
        self.checkbox('Upsample Faces', True)
        self.number('Upscale', 2, 1, 1, 4)
        self.number('Fidelity', 0.5, 0.01, 0.1, 1.0)
    def fn(self, data: dict) -> dict:
        if st.session_state.preview is not None:
            img = st.session_state.preview
        img = data.get('result_image', st.session_state.preview)

        if isinstance(img, list):
            img = img[0]

        args = {
            "image":img,
            "face_align":True,
            "background_enhance":True,
            "face_upsample":True,
            "upscale":4,
            "codeformer_fidelity":0.5,
        }

        from backend.codeformers import codeformersinference, init_codeformers

        images = [codeformersinference(**args)]

        data["result_image"] = images
        st.session_state.preview = images[0]
        st.session_state.images.append(images)
        if len(st.session_state['images']) > 8:
            if len(st.session_state['start_index']) < 8:
                st.session_state.start_index = 8
            else:
                st.session_state.start_index += 1
        return data


