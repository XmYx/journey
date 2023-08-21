import streamlit as st
def dynamic_controls(controls_config):
    values = {}
    expander = st.expander("Advanced Params")
    for control_name, config in controls_config.items():
        control_type = config['type']
        expose = config['expose']
        if expose:
            if control_type == 'text_input':
                values[control_name] = st.text_input(control_name, **config['params'])
            elif control_type == 'selectbox':
                values[control_name] = st.selectbox(control_name, **config['params'])
            elif control_type == 'number_input':
                values[control_name] = st.number_input(control_name, **config['params'])
            elif control_type == 'slider':
                values[control_name] = st.slider(control_name, **config['params'])
        else:
            with expander:
                if control_type == 'text_input':
                    values[control_name] = st.text_input(control_name, **config['params'])
                elif control_type == 'selectbox':
                    values[control_name] = st.selectbox(control_name, **config['params'])
                elif control_type == 'number_input':
                    values[control_name] = st.number_input(control_name, **config['params'])
                elif control_type == 'slider':
                    values[control_name] = st.slider(control_name, **config['params'])
    return values
