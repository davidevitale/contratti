@echo off
echo ============================================
echo ContractIQ - Shutdown Script
echo ============================================
echo.

echo Stopping all containers...
docker compose down

echo.
echo All services stopped.
echo.
echo To remove all data (database, vectors, etc.):
echo   docker compose down -v
echo.
pause
