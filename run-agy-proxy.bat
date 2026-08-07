@echo off
title Antigravity CLI Launcher with Proxy
powershell -NoProfile -ExecutionPolicy Bypass -Command "$proxy = Start-Process node -ArgumentList 'C:\Users\josh6\.gemini\antigravity-cli\model-proxy.js' -WindowStyle Minimized -PassThru; Write-Host 'Redirection proxy started in background.'; Start-Sleep -Seconds 2; $env:CLOUD_CODE_URL = 'http://127.0.0.1:8000'; Write-Host 'Redirection enabled (CLOUD_CODE_URL=http://127.0.0.1:8000)'; Write-Host 'Select Gemini 3.5 Flash (Medium) to use gemini-2.5-flash via AI Studio.'; Write-Host ''; agy; Stop-Process -Id $proxy.Id -Force"
