import streamlit as st
from PIL import Image, ImageOps
from rembg import remove

from modules.deforum_outpaint import load, outpaint

plugin_info = {"name": "RemBg"}


def plugin_tab(*args, **kwargs):
    col1, col2 = st.columns([2,5])
    with col1:
        with st.form('Bg Removal'):
            uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])
            prompt = st.text_area('Prompt')
            alpha_matting = st.toggle('Alpha Matting')
            alpha_matting_foreground_threshold = st.number_input('Foreground Threshold', value=240)
            alpha_matting_background_threshold = st.number_input('Background Threshold', value=10)
            alpha_matting_erode_size = st.number_input('Erode', value=10)
            # only_mask: bool = False
            post_process_mask = st.toggle('Post Process Mask')
            button = st.form_submit_button('Remove')

    with col2:
        if button:
            if uploaded_file is not None:
                mask = remove(Image.open(uploaded_file),
                                alpha_matting=alpha_matting,
                                alpha_matting_erode_size=alpha_matting_erode_size,
                                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                                alpha_matting_background_threshold=alpha_matting_background_threshold,
                                post_process_mask=post_process_mask,
                                only_mask=True)

                mask = ImageOps.invert(mask)

                input = Image.open(uploaded_file)

                # st.image(output)
                # st.image(uploaded_file)

                # # Ensure the image has an alpha layer
                # if output.mode != 'RGBA':
                #     raise Exception("The image is not in RGBA format")
                #
                # # Split the channels
                # r, g, b, a = output.split()
                #
                # # Merge R, G, B channels to get the color image
                # color_image = Image.merge("RGB", (r, g, b))
                # # The alpha channel becomes the 'mask'
                # mask_image = a

                load(model_repo='Lykon/dreamshaper-xl-1-0')

                args = {"prompt": prompt,
                        "image":input,
                        "mask":mask,
                        'seed': 0}

                image = outpaint(args)

                st.image(image)


