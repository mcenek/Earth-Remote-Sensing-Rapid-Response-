@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0GTM_Model0_OpenViewer.ps1" %*
if errorlevel 1 pause
