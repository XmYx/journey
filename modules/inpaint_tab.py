import sys
import numpy as np
import streamlit as st
from PIL import Image

from streamlit_drawable_canvas import st_canvas

from extras.sdjourney_backend import scheduler_type_values
from modules.outpaint_tab import do_inpaint

plugin_info = {"name": "DynaUI"}
MAX_SIZE = 1280




def plugin_tab(*args, **kwargs):

    global model_name
    model_name = "INPAINT"
    prompt = st.text_input('Prompt')
    scheduler_type = st.selectbox("Scheduler", scheduler_type_values)
    steps = st.number_input("Steps", min_value=1, max_value=1000, value=50)
    guidance_scale = st.number_input("Guidance Scale", min_value=0.0, max_value=25.0, value=6.5)
    strength = st.number_input("Strength", min_value=0.0, max_value=1.0, value=1.0)
    seed = st.text_input("Seed", value="0")

    image = st.file_uploader("Image", ["jpg", "png"])
    if image:
        image = Image.open(image)
        w, h = image.size
        # print(f"loaded input image of size ({w}, {h})")
        if max(w, h) > MAX_SIZE:
            factor = MAX_SIZE / max(w, h)
            w = int(factor * w)
            h = int(factor * h)
        width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        image = image.resize((width, height))
        print(f"resized to ({width}, {height})")
        col1, col2 = st.columns(2)
        rotate_left = col1.button('Rotate 90° CCW')
        rotate_right = col2.button('Rotate 90° CW')

        # Rotate the image if buttons are clicked
        if rotate_left:
            image = image.rotate(90, Image.NEAREST, expand = True)
        if rotate_right:
            image = image.rotate(-90, Image.NEAREST, expand = True)

        fill_color = "rgba(255, 255, 255, 0.0)"
        stroke_width = st.number_input("Brush Size",
                                       value=64,
                                       min_value=1,
                                       max_value=100)
        stroke_color = "rgba(255, 255, 255, 1.0)"
        bg_color = "rgba(0, 0, 0, 1.0)"
        drawing_mode = "freedraw"

        st.write("Canvas")
        st.caption(
            "Draw a mask to inpaint, then click the 'Send to Streamlit' button (bottom left, with an arrow on it).")
        canvas_result = st_canvas(
            fill_color=fill_color,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=image,
            update_streamlit=False,
            height=height,
            width=width,
            drawing_mode=drawing_mode,
            key="canvas",
        )
        if canvas_result:
            mask = canvas_result.image_data
            mask = mask[:, :, -1] > 0
            if mask.sum() > 0:
                mask = Image.fromarray(mask)

                result = [do_inpaint(prompt, image, mask, scheduler_type, steps, guidance_scale, strength, seed, "test")]
                st.write("Inpainted")
                for image in result:
                    st.image(image)


