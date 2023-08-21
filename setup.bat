@echo off
REM Check if venv directory exists
if not exist venv\Scripts\python.exe (
    REM If not, create the venv
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Install requirements
pip install -r requirements.txt

REM Run the main.py
python main.py
