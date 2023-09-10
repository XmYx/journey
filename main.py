import os
import subprocess
import sys
import time

from extras import singleton

def clone_if_not_exists(repo_url, local_path):
    """Clone the repository if the directory doesn't exist."""
    if not os.path.isdir(local_path):
        subprocess.run(["git", "clone", repo_url, local_path])

# Clone TokenFlow if not exists
clone_if_not_exists("https://github.com/XmYx/TokenFlow.git", "src/TokenFlow")

sys.path.append('CodeFormer')
sys.path.append('src/TokenFlow')


singleton.data = {}
singleton.data["models"] = {}
singleton.base_loaded = None

#singleton = Singleton.getInstance()

if __name__ == "__main__":

    process = subprocess.Popen(["streamlit", "run", "ui.py"])

    try:
        # Continuously check if the process is still running
        while process.poll() is None:
            time.sleep(1)  # wait for a second before checking again

    except KeyboardInterrupt:
        # Handle user interruption and terminate the subprocess if it's still running
        if process.poll() is None:
            process.terminate()