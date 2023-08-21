import base64
import json
import os
import streamlit as st

from extras.block_helpers import *
from main import singleton as gs

plugin_info = {"name": "Blocks"}


def initialize_session_state():
    if 'next_block_id' not in st.session_state:
        st.session_state.next_block_id = 0
    if 'looping' not in st.session_state:
        st.session_state.looping = False
    if "refresh" not in st.session_state:
        st.session_state.refresh = False


def dynamic_controls(controls_config, block_id):
    values = {}
    expander = st.expander("Advanced Params")
    for control_name, config in controls_config.items():
        control_type = config['type']
        expose = config.get('expose', True)
        unique_key = f"{block_id}_{control_name}"

        if control_type == "selectbox":
            options = config['params']['options']
            default_value = config['params'].get('value', options[0])
            default_index = options.index(default_value)
            location = st if expose else expander
            values[control_name] = location.selectbox(control_name, options, index=default_index, key=unique_key)
        else:
            location = st if expose else expander
            ui_function = getattr(location, control_type)
            values[control_name] = ui_function(control_name, **config['params'], key=unique_key)

    return values


def run_pipeline():
    results = []
    chained_data = {}
    for block in gs.data.get("pipeline", []):
        fn_name = blocks[block['name']]['fn']
        merged_values = {**chained_data, **block.get('values', {})}
        result = globals()[fn_name](merged_values)
        chained_data.update(result)
        results.append(result)
        if 'result_text' in result:
            st.write(result['result_text'])
        elif 'result_image' in result:
            gs.data["images"].append(result['result_image'])
            st.session_state.refresh = True

    return results
def plugin_tab():
    initialize_session_state()

    gs.data.setdefault("pipeline", [])
    gs.data.setdefault("images", [])

    show_buttons = st.sidebar.checkbox("Show Control Buttons", value=True)

    main_col_1, main_col_2 = st.columns(2)

    with main_col_1:
        # Display existing blocks and allow deletion
        for idx, block in enumerate(gs.data["pipeline"]):

            if show_buttons:
                col1, col2, col3, col4 = st.columns([5,1,1,1], gap="small")
            else:
                col1 = st.empty()

            with col1:
                #st.write(f"Configuring {block['name']}:")
                if 'controls' in blocks[block['name']]:
                    updated_values = dynamic_controls(blocks[block['name']]['controls'], block_id=block['block_id'])
                    block['values'].update(updated_values)  # Directly update the block's values
            if show_buttons:

                with col2:
                    if st.button(f"Delete {block['name']}", key=f"delete_{idx}"):
                        gs.data["pipeline"].pop(idx)
                        st.experimental_rerun()

                with col3:
                    # Disable "Up" button for the first block or for a starter block
                    disable_up = True if idx == 0 else False #or block['category'] == 'starter' else False
                    if st.button("↑", key=f"up_{idx}", disabled=disable_up) and idx > 0:
                        gs.data["pipeline"][idx], gs.data["pipeline"][idx-1] = gs.data["pipeline"][idx-1], gs.data["pipeline"][idx]
                        st.experimental_rerun()

                with col4:
                    # Disable "Down" button for the last block or if current block or next block is an 'end' block
                    disable_down = True if idx == len(gs.data["pipeline"]) - 1 or block['category'] == 'end' or (idx + 1 < len(gs.data["pipeline"]) and gs.data["pipeline"][idx+1]['category'] == 'end') else False
                    if st.button("↓", key=f"down_{idx}", disabled=disable_down) and idx < len(gs.data["pipeline"]) - 1:
                        gs.data["pipeline"][idx], gs.data["pipeline"][idx+1] = gs.data["pipeline"][idx+1], gs.data["pipeline"][idx]
                        st.experimental_rerun()


    with main_col_2:
        # Display all stored images in the third column
        if gs.data["images"] is not None:
            display = gs.data["images"][::-1]  # Reverse the list using slicing
            for img in display:
                st.image(img)
    if st.session_state.refresh:
        st.session_state.refresh = False
        st.experimental_rerun()

    with st.sidebar:

        filename = st.text_input("Filename for saving", "pipeline.json")
        if st.button('Save Pipeline'):
            save_path = os.path.join('pipelines', filename)
            with open(save_path, 'w') as f:
                json.dump(gs.data["pipeline"], f)
            st.write(f"Saved to {save_path}")


        # List all JSON files in the 'pipelines' folder
        pipeline_files = [f for f in os.listdir('pipelines') if
                          os.path.isfile(os.path.join('pipelines', f)) and f.endswith('.json')]
        selected_file = st.selectbox("Select a pipeline to load", ["Choose a file..."] + pipeline_files)

        # Load the selected pipeline
        if selected_file != "Choose a file...":
            with open(os.path.join('pipelines', selected_file), 'r') as f:
                pipeline_data = json.load(f)
                gs.data["pipeline"] = pipeline_data
                st.session_state.refresh = True
                st.experimental_rerun()

        # Upload a pipeline file
        uploaded_file = st.file_uploader("Or upload a Pipeline JSON", type="json")
        if uploaded_file:
            pipeline_data = json.loads(uploaded_file.getvalue())
            gs.data["pipeline"] = pipeline_data
            st.session_state.refresh = True

        # Display available blocks based on the last block's category
        if not gs.data["pipeline"]:
            available_blocks = [name for name, block in blocks.items() if block['category'] == 'starter']
        else:
            last_block = gs.data["pipeline"][-1]
            block_controls = blocks[last_block['name']]['controls']
            for control_name, config in block_controls.items():
                default_value = last_block['values'].get(control_name, config['params'].get('value', None))
                config['params']['value'] = default_value

            if last_block['category'] == 'starter':
                available_blocks = [name for name, block in blocks.items() if block['category'] in ['middle', 'end']]
            elif last_block['category'] == 'middle':
                available_blocks = [name for name, block in blocks.items() if block['category'] in ['middle', 'end']]
            else:
                st.write("Pipeline completed. Reset to add new blocks.")
                available_blocks = []

            for control_name, config in block_controls.items():
                if config['type'] == 'selectbox':
                    options = config['params']['options']
                    default_value = last_block['values'].get(control_name, options[0])
                    config['params']['index'] = options.index(default_value)
                else:
                    default_value = last_block['values'].get(control_name, config['params'].get('value', None))
                    config['params']['value'] = default_value


        if available_blocks:
            selected_block = st.selectbox("Select a block to add", available_blocks)
            # if 'controls' in blocks[selected_block]:
            #     block_values = dynamic_controls(blocks[selected_block]['controls'])
            # else:
            #     block_values = {}
            if st.button(f"Add {selected_block}"):
                gs.data["pipeline"].append({
                    'name': selected_block,
                    'category': blocks[selected_block]['category'],
                    'values': {},
                    'block_id': st.session_state.next_block_id  # Assign a unique block_id
                })
                st.session_state.next_block_id += 1  # Increment the counter
                st.experimental_rerun()


        if st.button("Clear Pipeline", key="clear_pipeline_blocks"):
            gs.data["pipeline"].clear()
            st.experimental_rerun()
        if st.button("Clear Images", key="clear_images_blocks"):
            gs.data["images"].clear()
            st.experimental_rerun()

    with main_col_1:
        loop = st.sidebar.checkbox('Loop')
        run_btn = st.sidebar.button("Run Pipeline")

        if run_btn:
            results = run_pipeline()
            st.experimental_rerun()