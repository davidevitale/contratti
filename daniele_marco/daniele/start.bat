@echo off
echo ============================================
echo ContractIQ - Startup Script
echo ============================================
echo.

echo Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not in PATH
    echo Please install Docker Desktop: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

echo Docker is installed.
echo.

echo Stopping any existing containers...
docker compose down >nul 2>&1

echo.
echo Building and starting all services...
echo This may take 5-10 minutes on first run.
echo.

docker compose up --build -d

echo.
echo ============================================
echo Waiting for services to start...
echo ============================================
echo.

timeout /t 10 /nobreak >nul

echo Checking service health...
echo.

curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo WARNING: Gateway not ready yet. Waiting...
    timeout /t 15 /nobreak >nul
)

curl -s http://localhost:8001/health >nul 2>&1
if errorlevel 1 (
    echo WARNING: DSPy Agents not ready yet. Waiting...
    timeout /t 15 /nobreak >nul
)

curl -s http://localhost:8002/health >nul 2>&1
if errorlevel 1 (
    echo WARNING: RAG Service not ready yet. Waiting...
    timeout /t 15 /nobreak >nul
)

echo.
echo ============================================
echo ContractIQ is starting!
echo ============================================
echo.
echo Frontend:     http://localhost:3000
echo API Gateway:  http://localhost:8000
echo API Docs:     http://localhost:8000/docs
echo.
echo Services may take a few minutes to fully initialize.
echo.
echo To view logs: docker compose logs -f
echo To stop:      docker compose down
echo.
echo Press any key to view service status...
pause >nul

docker compose ps

echo.
echo Done!
