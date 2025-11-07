@echo off
echo ========================================
echo ML Comparison Toolkit - All Demos
echo ========================================
echo.
echo This will open 4 Streamlit demos:
echo 1. Supervised Learning (Port 8501)
echo 2. Ensemble Learning (Port 8502)
echo 3. Reinforcement Learning (Port 8503)
echo 4. Semi-Supervised Learning (Port 8504)
echo.
echo Press Ctrl+C in each window to stop
echo ========================================
echo.

start "Supervised Learning" cmd /k "streamlit run supervised\streamlit_supervised_demo.py --server.port 8501"
timeout /t 3 /nobreak >nul

start "Ensemble Learning" cmd /k "streamlit run ensemble_learning\streamlit_el_demo.py --server.port 8502"
timeout /t 3 /nobreak >nul

start "Reinforcement Learning" cmd /k "streamlit run reinforcement_learning\streamlit_rl_demo.py --server.port 8503"
timeout /t 3 /nobreak >nul

start "Semi-Supervised Learning" cmd /k "streamlit run semi_supervised_learning\streamlit_semi_supervised_demo.py --server.port 8504"

echo.
echo All demos are starting...
echo Check your browser for the following URLs:
echo - Supervised: http://localhost:8501
echo - Ensemble: http://localhost:8502
echo - Reinforcement: http://localhost:8503
echo - Semi-Supervised: http://localhost:8504
echo.
pause
