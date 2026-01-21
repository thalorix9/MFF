@echo off
call conda activate mff_env
if %errorlevel% neq 0 (
    echo Error: Could not activate environment 'mff_env'.
    echo Please run setup_env.bat first.
    pause
    exit /b
)

echo Running MFF pipeline...
python main.py
pause