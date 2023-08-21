import streamlit as st
plugin_info = {"name": "Blocks"}

blocks = {
    "StarterBlock": {
        "category": "starter",
        "controls": {
            "starter_text": {
                "type": "text_input",
                "expose": True,
                "params": {"value": "Starter Block"}
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
    }
}

def starter_fn(starter_text):
    return starter_text

def middle_fn(middle_text):
    return middle_text

def end_fn(end_text):
    return end_text

def dynamic_controls(controls_config):
    values = {}
    expander = st.expander("Advanced Params")
    for control_name, config in controls_config.items():
        control_type = config['type']
        expose = config['expose']
        if expose:
            ui_function = getattr(st, control_type)
            values[control_name] = ui_function(control_name, **config['params'])
        else:
            with expander:
                ui_function = getattr(st, control_type)
                values[control_name] = ui_function(control_name, **config['params'])
    return values
def display_block_ui(block_name, block_dict):
    # Store block configuration values
    values = {}
    for control in block_dict['ui']:
        if control['type'] == 'text_input':
            values[control['name']] = st.text_input(control['name'], control['default'])
    return values
def plugin_tab():
    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = []

    # Display existing blocks and allow deletion
    for idx, block in enumerate(st.session_state.pipeline):
        st.write(f"{block['name']}: {block['values']}")
        if st.button(f"Delete {block['name']}", key=f"delete_{idx}"):
            st.session_state.pipeline.pop(idx)

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
        block_values = dynamic_controls(blocks[selected_block]['controls'])
        if st.button(f"Add {selected_block}"):
            st.session_state.pipeline.append({
                'name': selected_block,
                'category': blocks[selected_block]['category'],
                'values': block_values
            })
            st.experimental_rerun()


    # Execute all blocks in sequence
    gn_btn = st.button("G")
    if gn_btn:
        results = []
        for block in st.session_state.pipeline:
            fn_name = blocks[block['name']]['fn']
            result = globals()[fn_name](**block['values'])
            results.append(result)
        st.write(results)