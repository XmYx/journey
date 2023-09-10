# main_app.py
import subprocess

import streamlit as st
st.set_page_config(layout="wide")
import os
import importlib.util

from main import singleton as gs

import os
import json


def load_config_and_initialize():
    # Load the JSON file
    with open("config/defaults.json", "r") as f:
        config = json.load(f)

    # Create the default folders if they don't exist
    for key, folder in config['folders'].items():
        os.makedirs(folder, exist_ok=True)

    # Store the loaded configuration in gs.data
    if "config" not in gs.data:
        gs.data["config"] = {}
    gs.data["config"].update(config)

    return config


config = load_config_and_initialize()
print(gs.data['config'])

if "models" not in gs.data:
    print("Instantiating models dictionary in singleton")
    gs.data["models"] = {}


# Function to import a module from a file path
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
def save_tabs_to_json():
    with open('config/tabs.json', 'w') as f:
        json.dump({key: value["active"] for key, value in st.session_state.modules.items()}, f)
def load_tabs_from_json():
    try:
        with open('config/tabs.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def main():

    # Load the tabs from the JSON file if it exists
    json_filepath = 'config/tabs.json'


    if "modules" not in st.session_state:
        module_files = [f for f in os.listdir('modules') if f.endswith('.py') and f != '__init__.py']
        st.session_state.modules = {}
        tab_names = []

        for file in module_files:
            module_name = file.replace('.py', '')
            module = import_module_from_path(module_name, os.path.join('modules', file))

            st.session_state.modules[module_name] = {   "name":module.plugin_info["name"],
                                                        "module":module,
                                                        "active":True
                                                        }
            tab_names.append(module.plugin_info["name"])
        st.session_state.modules["toggle_tab"] = {   "name":"Toggle Tabs",
                                                        "module":None,
                                                        "active":True
                                                        }
        tab_names.append("Toggle Tabs")
        st.session_state.tab_names = tab_names

    if "active_tabs" not in st.session_state:
        st.session_state.active_tabs = st.session_state.modules

    active_tabs_from_json = load_tabs_from_json()
    for module_name, is_active in active_tabs_from_json.items():
        if module_name in st.session_state.modules:
            st.session_state.modules[module_name]["active"] = is_active

    #tabs = st.tabs(st.session_state.tab_names)
    tabs = st.tabs([value["name"] for key, value in st.session_state.modules.items() if value["active"]])

    with tabs[len(tabs) - 1]:
        with st.form("Toggle Tabs"):
            toggles = {}
            for key, value in st.session_state.modules.items():
                if value["name"] != "Toggle Tabs":
                    value["active"] = st.toggle(f'Enable {value["name"]} tab', value=value["active"])

            if st.form_submit_button("Set Active Tabs"):
                save_tabs_to_json()

                st.experimental_rerun()


    active_modules = {}
    x = 0
    for key, value in st.session_state.modules.items():
        if value["active"]:
            if value["name"] != "Toggle Tabs":
                active_modules[key] = value
                with tabs[x]:
                    value["module"].plugin_tab(None, None)
                x += 1












def toggle_tab():
    with st.form('toggle'):
        modules = {}

        for tab, module in st.session_state.modules.items():

            print("tab, module", tab, module)
            if not isinstance(module, str):
                modules[tab] = st.toggle(f"Enable {module.plugin_info['name']}")

        if st.form_submit_button('Update Tabs'):

            print(st.session_state.modules)


            # Update the active modules in session state
            # active_modules = [tab for tab, is_active in modules.items() if is_active]
            # st.session_state.active_modules = active_modules
            #
            # # Save the active modules to json
            # save_tabs_to_json(active_modules)
            #
            # st.experimental_rerun()





def main__():
    module_files = [f for f in os.listdir('modules') if f.endswith('.py') and f != '__init__.py']

    modules = {}
    tab_names = []

    for file in module_files:
        module_name = file.replace('.py', '')
        module = import_module_from_path(module_name, os.path.join('modules', file))
        modules[module_name] = module

        tab_names.append(module.plugin_info["name"])

    st.session_state.tab_names = tab_names
    st.session_state.modules = modules

    modules["TabToggles"] = "tab_toggle"
    tab_names.append("TabToggles")

    # Load active modules from json if exists
    if not hasattr(st.session_state, 'active_modules'):
        st.session_state.active_modules = load_tabs_from_json()

    # Ensure TabToggles is always present

    use_tabs = st.sidebar.toggle('Use Tabs', value=True)
    if use_tabs:
        active_tabs = st.session_state.active_modules if st.session_state.active_modules else modules
        tabs = st.tabs(active_tabs)
        x = 0

        for module_name in active_tabs:
            if module_name in modules:
                with tabs[x]:
                    if module_name != "TabToggles":
                        print(module_name, x, tab_names)
                        modules[module_name].plugin_tab(x, tab_names)
                    else:
                        toggle_tab()
            else:
                print(f"Module {module_name} not found in loaded modules!")
            x += 1
    else:
        # Display buttons on the sidebar for each module
        selected_module = st.sidebar.radio("Choose a module", list(modules.keys()))
        x = 1
        # Load the relevant module in the main section based on the selected button
        if selected_module in modules:
            modules[selected_module].plugin_tab(x, tab_names)

if __name__ == '__main__':
    main()