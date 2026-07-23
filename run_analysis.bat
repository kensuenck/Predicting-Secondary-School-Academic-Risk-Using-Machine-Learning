@echo off
python src\dissertation_analysis.py --data data\student-mat.csv --output-dir results_local
if errorlevel 1 exit /b %errorlevel%
echo Analysis completed. See results_local\
