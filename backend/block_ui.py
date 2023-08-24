import streamlit as st

from backend.block_base import save_to_json, load_from_json, AVAILABLE_BLOCKS
from main import singleton as gs

def run_pipeline(start_index=0):
    cumulative_data = {}
    for block in gs.data['added_blocks'][start_index:]:
        cumulative_data = block.fn(cumulative_data)
    return cumulative_data
def initialize():
    if "added_blocks" not in gs.data:
        gs.data["added_blocks"] = []
    if 'images' not in st.session_state:
        st.session_state.images = []
def display_sidebar():
    # Sidebar UI
    st.sidebar.title("Add Blocks")
    selected_block_name = st.sidebar.selectbox("Choose a block:", [block.__name__ for block in AVAILABLE_BLOCKS])

    if st.sidebar.button("Add Block"):
        # Create an instance of the selected block and add it to the session state
        selected_block_class = next((block for block in AVAILABLE_BLOCKS if block.__name__ == selected_block_name),
                                    None)
        if selected_block_class:
            gs.data['added_blocks'].append(selected_block_class())
    if st.sidebar.button("Save to JSON"):
        save_to_json("pipeline.json")
        st.write("Saved pipeline to pipeline.json")

    if st.sidebar.button("Load from JSON"):
        load_from_json("pipeline.json")
        #gs.data['added_blocks'] = block_holder.blocks
        st.write("Loaded pipeline from pipeline.json")
        st.experimental_rerun()

    global hide
    hide = st.sidebar.checkbox('Hide Buttons')

def render_widget(widget):
    """Function to render a widget based on its type."""

    if widget.widget_type == "text":
        widget.value = st.text_input(widget.name, value=widget.value, key=widget.uid)
    elif widget.widget_type == "text_multi":
        widget.value = st.text_area(widget.name, value=widget.value, key=widget.uid)
    elif widget.widget_type == "number":
        widget.value = st.number_input(widget.name, value=widget.value, key=widget.uid)
    elif widget.widget_type == "dropdown":

        selection = st.selectbox(widget.name, options=widget.options,
                                 index=widget.selected_index, key=widget.uid)
        widget.selected_index = widget.options.index(selection)
    elif widget.widget_type == "slider":
        widget.value = st.slider(widget.name, min_value=0, max_value=100, value=widget.value, key=widget.uid)
    elif widget.widget_type == "checkbox":
        widget.value = st.checkbox(widget.name, value=widget.value, key=widget.uid)
    elif widget.widget_type == "multiselect":
        widget.value = st.multiselect(widget.name, options=widget.options, default=widget.value, key=widget.uid)


def display_nav_buttons(col2, col3, col4, col5, index, block):
    # Create buttons for moving up, moving down, and deleting
    move_up = False if index == 0 else True  # Disable for the first block
    move_down = False if index == len(gs.data['added_blocks']) - 1 else True  # Disable for the last block

    # Using columns to make the buttons more compact

    with col2:
        if move_up:
            # Use Unicode arrow up character for the move up button
            if st.button("↑", key=f"up_{block.uid}", help="Move block up"):
                # Swap blocks
                gs.data['added_blocks'][index], gs.data['added_blocks'][index - 1] = \
                    gs.data['added_blocks'][index - 1], gs.data['added_blocks'][index]
                st.experimental_rerun()

    with col3:
        if move_down:
            # Use Unicode arrow down character for the move down button
            if st.button("↓", key=f"down_{block.uid}", help="Move block down"):
                # Swap blocks
                gs.data['added_blocks'][index], gs.data['added_blocks'][index + 1] = \
                    gs.data['added_blocks'][index + 1], gs.data['added_blocks'][index]
                st.experimental_rerun()

    with col4:
        # Use Unicode cross mark character for the delete button
        if st.button("✖", key=f"delete_{block.uid}", help="Delete block"):
            del gs.data['added_blocks'][index]
            st.experimental_rerun()
    with col5:
        if st.button("▶", key=f"run_from_{block.uid}"):
            run_pipeline(start_index=index)
def display_main_button(col):
    with col:
        if st.button("Run Blocks"):
            result = run_pipeline()


def display_block_with_controls(block, index, main_col_1):
    """Display a block's widgets along with its control buttons."""

    block.index = index

    with main_col_1:
        advanced = st.expander("Advanced")
        # Use container for the box
        with st.expander(block.name, expanded=True):



            if not hide:
                col1, col2, col3, col4, col5 = st.columns([10, 1, 1, 1, 1])
            else:
                col1 = main_col_1

            # Display block's widgets in col1
            with col1:

                # Display widgets without 'expose=False' directly
                for widget in block.widgets:
                    if not hasattr(widget, 'expose') or widget.expose:
                        render_widget(widget)

            # Display widgets with 'expose=False' inside an expander
            with advanced:
                for widget in block.widgets:
                    if hasattr(widget, 'expose') and not widget.expose:
                        render_widget(widget)

            # Display block's control buttons in col2
            if not hide:
                display_nav_buttons(col2, col3, col4, col5, index, block)
