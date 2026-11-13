@echo off
echo ========================================================
echo  Uploading Credit Card Fraud Detection Platform to GitHub
echo ========================================================
cd /d "%~dp0"
git push -u origin main
echo ========================================================
echo  Upload completed successfully!
echo ========================================================
pause
