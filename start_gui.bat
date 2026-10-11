@echo off
chcp 65001 >nul
cd /d "%~dp0"
title LucidChaos 控制台启动器

echo ============================================================
echo   正在启动《卡厄思梦境》LucidChaos 现代化控制台客户端...
echo ============================================================

python run_gui.py

if errorlevel 1 (
    echo.
    echo [提示] 客户端已退出或发生错误。
    pause
)
