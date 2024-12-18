@echo off

set VENV_DIR=venv

python -m venv %VENV_DIR%

call %VENV_DIR%\Scripts\activate

pip install -r requirements.txt

powershell -command "Expand-Archive -Path 'example_dataset.zip' -DestinationPath '.' -Force"

deactivate