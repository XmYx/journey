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
    def __init__(self):
        super().__init__()
        self.text('prompt', multiline=True)

    def fn(self, data: dict) -> dict:
        data["prompt"] = self.widgets[0].value
        print(data)

        return data
@register_class
class DiffusersPromptStyleBlock(BaseBlock):
    def __init__(self):
        super().__init__()
        self.dropdown('prompt', style_keys)

    def fn(self, data: dict) -> dict:
        data["style"] = self.widgets[0].options[self.widgets[0].selected_index]
        data = style(data)
        return data
@register_class
class DiffusersSamplerBlock(BaseBlock):
    def __init__(self):
        super().__init__()
        self.dropdown('Scheduler', scheduler_type_values)
    def fn(self, data: dict) -> dict:

        target_device = "cuda"
        if gs.data["models"]["base"].device.type != target_device:
            gs.data["models"]["base"].to(target_device)
        widget = self.widgets[0]
        data['scheduler'] = widget.options[widget.selected_index]
        args, pipe = check_args(data, gs.data['models']['base'])
        result = pipe.generate(**args)
        gs.data["latents"] = result[0]
        data["result_image"] = result[1]

        st.session_state.preview = result[1][0]

        if "images" not in st.session_state:
            st.session_state.images = []

        st.session_state.images.append(result[1])

        return data
@register_class
class DiffusersRefinerBlock(BaseBlock):
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

        return data


