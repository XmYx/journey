import base64
import json

import streamlit as st
from main import singleton as gs


plugin_info = {"name": "Blocks"}

blocks = {
    "StarterBlock": {
        "category": "starter",
        "controls": {
            "prompt": {
                "type": "text_input",
                "expose": True,
                "params": {"value": "Starter Block"}
            },
            "num_inference_steps": {
                "type": "slider",
                "expose": True,
                "params": {"value": 25}
            }
        },
        "fn": "starter_fn"
    },
    "MiddleBlock": {
        "category": "middle",
        "controls": {
            "middle_text": {
                "type": "text_input",
                "expose": False,
                "params": {"value": "Middle Block"}
            }
        },
        "fn": "middle_fn"
    },
    "EndBlock": {
        "category": "end",
        "controls": {
            "end_text": {
                "type": "text_input",
                "expose": True,
                "params": {"value": "End Block"}
            }
        },
        "fn": "end_fn"
    },
    "Generate": {
        "category": "end",
        "fn": "generate"
    }
}

def generate(args):
    target_device = "cuda"
    if gs.data["models"]["base"].device.type != target_device:
        gs.data["models"]["base"].to(target_device)

    gen_args = {"prompt":"test",
                "num_inference_steps":10}

    if "prompt" in args:
        gen_args["prompt"] = args["prompt"]
    if "num_inference_steps" in args:
        gen_args["num_inference_steps"] = args["num_inference_steps"]

    result = gs.data["models"]["base"](**gen_args)

    result_dict = {"result_image":result.images[0]}

    result_dict = {**args, **result_dict}
    return result_dict



def starter_fn(starter_text):

    print("starter module data", starter_text)

    return starter_text

def middle_fn(middle_text):
    print(middle_text)
    return middle_text

def end_fn(end_text):
    return end_text

def dynamic_controls(controls_config, block_idx=None):
    values = {}
    expander = st.expander("Advanced Params")
    for control_name, config in controls_config.items():
        control_type = config['type']
        expose = config['expose']
        unique_key = f"{block_idx}_{control_name}" if block_idx is not None else control_name
        if expose:
            ui_function = getattr(st, control_type)
            values[control_name] = ui_function(control_name, **config['params'], key=unique_key)
        else:
            with expander:
                ui_function = getattr(st, control_type)
                values[control_name] = ui_function(control_name, **config['params'], key=unique_key)
    return values
def plugin_tab():
    refresh = False
    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = []
        st.session_state.images = []  # To store images returned by blocks
        st.session_state.refresh = False

    main_col_1, main_col_2 = st.columns(2)

    with main_col_1:
        # Display existing blocks and allow deletion
        for idx, block in enumerate(st.session_state.pipeline):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"Configuring {block['name']}:")
                if 'controls' in blocks[block['name']]:
                    updated_values = dynamic_controls(blocks[block['name']]['controls'], block_idx=idx)
                    block['values'].update(updated_values)  # Directly update the block's values

            with col2:
                if st.button(f"Delete {block['name']}", key=f"delete_{idx}"):
                    st.session_state.pipeline.pop(idx)
                    st.experimental_rerun()

    with main_col_2:
        # Display all stored images in the third column
        if st.session_state.images is not None:
            display = st.session_state.images[::-1]  # Reverse the list using slicing
            for img in display:
                st.image(img)
    if st.session_state.refresh:
        st.session_state.refresh = False
        st.experimental_rerun()

    with st.sidebar:

        # Save Pipeline
        if st.button('Save Pipeline'):
            pipeline_json = json.dumps(st.session_state.pipeline)
            b64 = base64.b64encode(pipeline_json.encode()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="pipeline.json">Download Pipeline JSON</a>'
            st.markdown(href, unsafe_allow_html=True)

        # Load Pipeline
        uploaded_file = st.file_uploader("Upload Pipeline JSON", type="json")
        if uploaded_file:
            pipeline_data = json.loads(uploaded_file.getvalue())
            st.session_state.pipeline = pipeline_data
            st.session_state.refresh = True

        # Display available blocks based on the last block's category
        if not st.session_state.pipeline:
            available_blocks = [name for name, block in blocks.items() if block['category'] == 'starter']
        elif st.session_state.pipeline[-1]['category'] == 'starter':
            available_blocks = [name for name, block in blocks.items() if block['category'] in ['middle', 'end']]
        elif st.session_state.pipeline[-1]['category'] == 'middle':
            available_blocks = [name for name, block in blocks.items() if block['category'] in ['middle', 'end']]
        else:
            st.write("Pipeline completed. Reset to add new blocks.")
            available_blocks = []
        if available_blocks:
            selected_block = st.selectbox("Select a block to add", available_blocks)
            if 'controls' in blocks[selected_block]:
                block_values = dynamic_controls(blocks[selected_block]['controls'])
            else:
                block_values = {}
            if st.button(f"Add {selected_block}"):
                st.session_state.pipeline.append({
                    'name': selected_block,
                    'category': blocks[selected_block]['category'],
                    'values': block_values
                })
                st.experimental_rerun()
    with main_col_1:
        gn_btn = st.button("Generate")
    if gn_btn:
        results = []
        chained_data = {}  # This will accumulate data across blocks
        for block in st.session_state.pipeline:
            fn_name = blocks[block['name']]['fn']
            # Merge the accumulated data with the current block's values
            merged_values = {**chained_data, **block['values']}

            result = globals()[fn_name](merged_values)

            # Accumulate the result for the next iteration
            chained_data.update(result)

            results.append(result)

            # Display the result based on its type
            if 'result_text' in result:
                st.write(result['result_text'])
            elif 'result_image' in result:
                #st.image(result['result_image'])
                st.session_state.images.append(result['result_image'])  # Store the image in session_state
                st.session_state.refresh = True

        st.write("All results:", results)
        st.experimental_rerun()