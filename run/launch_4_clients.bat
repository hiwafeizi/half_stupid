@echo off
echo ============================================
echo   Launching 4 Malmo Minecraft Clients
echo ============================================
echo.

set MALMO_DIR=C:\Users\hiwa\Malmo_Python3.7\Minecraft

echo Starting client on port 10000...
start "Malmo-10000" cmd /c "cd /d %MALMO_DIR% && launchClient.bat --port 10000"
timeout /t 30 /nobreak >nul

echo Starting client on port 10001...
start "Malmo-10001" cmd /c "cd /d %MALMO_DIR% && launchClient.bat --port 10001"
timeout /t 30 /nobreak >nul

echo Starting client on port 10002...
start "Malmo-10002" cmd /c "cd /d %MALMO_DIR% && launchClient.bat --port 10002"
timeout /t 30 /nobreak >nul

echo Starting client on port 10003...
start "Malmo-10003" cmd /c "cd /d %MALMO_DIR% && launchClient.bat --port 10003"
timeout /t 30 /nobreak >nul

echo.
echo ============================================
echo   All 4 clients launching!
echo   Wait for all to show DORMANT, then run:
echo.
echo   conda activate malmo
echo   python run/multi_agent_world.py
echo ============================================
pause
