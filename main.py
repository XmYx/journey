import subprocess
import time

from extras.singleton import Singleton

singleton = Singleton.getInstance()

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