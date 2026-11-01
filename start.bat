@echo off
cd /d "%~dp0"
echo ====================================================
echo  InsureAI - Intelligent Insurance Claim Prediction
echo  Real World Vehicle Insurance Dataset
echo ====================================================

echo.
echo [1/4] Checking Python environment...
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Project virtual environment not found.
    echo Run: py -3.10 -m venv .venv
    echo Then: .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import uvicorn, joblib" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Backend dependencies are missing from the project virtual environment.
    echo Run: .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [2/4] Checking model files...
if not exist models\current.json (
    echo      Active Champion model pointer not found.
    echo      Training on REAL dataset...
    echo      This may take 3-5 minutes on first run.
    set PYTHONIOENCODING=utf-8
    .venv\Scripts\python.exe -m backend.ml.train_real
    if %errorlevel% neq 0 (
        echo [ERROR] Real data training failed. Please inspect logs.
        pause
        exit /b 1
    )
) else (
    echo      Authoritative Champion model found in Model Registry.
    echo      To retrain on real data: python -m backend.ml.train_real
)

echo.
echo [3/4] Starting FastAPI backend on http://localhost:8000 ...
set PYTHONIOENCODING=utf-8
set LOKY_MAX_CPU_COUNT=4
start "InsureAI Backend" ".venv\Scripts\python.exe" -m uvicorn backend.main:app --reload --port 8000

echo.
echo      Waiting 4 seconds for backend to initialize...
timeout /t 4 /nobreak >nul

echo.
echo [4/4] Starting React frontend on http://localhost:5173 ...
start "InsureAI Frontend" cmd /k "cd /d %~dp0\frontend && npm run dev"

echo.
echo ====================================================
echo  InsureAI is running!
echo.
echo  Frontend:   http://localhost:5173
echo  Backend:    http://localhost:8000
echo  API Docs:   http://localhost:8000/docs
echo  Data Mode:  http://localhost:8000/data-mode
echo.
echo  Dataset: Real Vehicle Insurance (58,592 records)
echo  Model trained with SMOTE + 5-fold CV
echo ====================================================
echo.
pause
