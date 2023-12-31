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

# Function to import a module from a file path
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
def main():
    # List all python files in the 'modules' folder
    module_files = [f for f in os.listdir('modules') if f.endswith('.py') and f != '__init__.py']

    # Import the modules
    modules = {}
    tab_names = []
    for file in module_files:
        module_name = file.replace('.py', '')
        module = import_module_from_path(module_name, os.path.join('modules', file))
        modules[module_name] = module
        tab_names.append(module.plugin_info["name"])

    # Streamlit app

    #st.title("Dynamic Tabs from Modules")

    # Create tabs for each module
    tabs = st.tabs(tab_names)

    for module_name, tab in zip(modules.keys(), tabs):
        with tab:
            modules[module_name].plugin_tab()

main()

