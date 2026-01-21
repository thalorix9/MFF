@echo off
echo Creating Conda environment 'mff_env' from environment.yml...
call conda env create -f environment.yml

echo.
echo Environment created successfully!
echo To activate this environment, run:
echo     conda activate mff_env
echo.
pause