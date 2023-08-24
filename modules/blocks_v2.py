import streamlit as st

from backend.block_ui import initialize, display_sidebar, display_block_with_controls, display_main_button
from main import singleton as gs

plugin_info = {"name": "Blocks v2"}

from backend.block_classes import *
def display_preview():
    if 'preview' in st.session_state:
        if st.session_state.preview is not None:
            st.session_state.preview_holder = st.image(st.session_state.preview)
        else:
            st.session_state.preview_holder = st.empty()
def plugin_tab():
    initialize()

    display_sidebar()

    col1, col2 = st.columns(2)
    if "preview_holder" not in st.session_state:
        with col2:
            st.session_state.preview_holder = st.empty()

    display_main_button(col1)
    # Display each block with its control buttons
    for index, block in enumerate(gs.data['added_blocks']):
        display_block_with_controls(block, index, col1)

    with col2:
        display_preview()

    st.divider()

    # Determine the number of columns (up to 8)
    num_images = len(st.session_state.images)
    num_columns = min(num_images, 6)

    # If there's no start_index in the session state, initialize it to 0
    if "start_index" not in st.session_state:
        st.session_state.start_index = 0
    buttons = {}
    if num_columns > 0:
        # Create the columns
        image_columns = st.columns(num_columns)

        # Display the images in the columns
        for idx, col in enumerate(image_columns):
            if idx + st.session_state.start_index < num_images:
                col.image(st.session_state.images[
                              idx + st.session_state.start_index], width=256)  # Replace with your method of displaying images
                with col:
                    buttons[idx] = st.button("View Image", key=f"preview_button_{idx}")

                    if buttons[idx]:
                        st.session_state.preview_holder.image(st.session_state.images[
                                  idx + st.session_state.start_index])
        # Step through images
        left, _, right = st.columns([1, 6, 1])
        if left.button("←") and st.session_state.start_index > 0:
            st.session_state.start_index -= 1  # Move the images to the left

        if right.button("→") and st.session_state.start_index + num_columns < num_images:
            st.session_state.start_index += 1  # Move the images to the right

        st.write("test")