@echo off
chcp 936 >nul
cd /d "%~dp0.."
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8

echo ============================================
echo   缺陷检测平台 - 静态图 (Faster R-CNN + SAHI)
echo ============================================
echo.

rem 选择解释器: 优先环境变量 PLATFORM_PY, 否则找第一个同时装了 torch 和 PyQt5 的
if defined PLATFORM_PY goto HAVE_PY

call :TRY "D:/Python3.14/python.exe"
if defined PYEXE goto HAVE_PY
call :TRY "python"
if defined PYEXE goto HAVE_PY
call :TRY "py"
if defined PYEXE goto HAVE_PY

echo [启动失败] 没有找到同时装有 torch 和 PyQt5 的 Python。
echo.
echo 解决办法:
echo   1) 安装依赖: pip install -r requirements.txt
echo      (torch 要走 CUDA 官方源, 见 requirements.txt 头部注释)
echo   2) 或设置环境变量 PLATFORM_PY 指向正确的 python.exe:
echo        set PLATFORM_PY=D:/Python3.14/python.exe
echo   3) 只想先验证效果(不开界面):
echo        python defect_platform\smoke_test.py --n 1
echo.
pause
exit /b 1

:HAVE_PY
if defined PLATFORM_PY set "PYEXE=%PLATFORM_PY%"
echo 使用解释器: %PYEXE%
echo 工作目录  : %CD%
echo.
echo 正在启动界面 (首次加载模型约 1-3 秒)...
%PYEXE% defect_platform\platform_ui.py %*
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
  echo.
  echo [异常退出] 退出码 %RC%
  echo 常见原因:
  echo   1. checkpoint 不存在 (检查 config.default_ckpt)
  echo   2. 显存不足 / 依赖不全
  echo   3. 想看完整报错, 改用命令行启动:
  echo      %PYEXE% defect_platform\platform_ui.py
  echo.
  pause
)
endlocal
exit /b 0

:TRY
set "_C=%~1"
%_C% -c "import torch, PyQt5" >nul 2>nul
if not errorlevel 1 (
  set "PYEXE=%_C%"
  echo [OK] 找到可用解释器: %_C%
)
exit /b 0
