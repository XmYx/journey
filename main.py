import subprocess
from extras.singleton import Singleton

singleton = Singleton.getInstance()

if __name__ == "__main__":
    subprocess.Popen(["streamlit", "run", "ui.py"])