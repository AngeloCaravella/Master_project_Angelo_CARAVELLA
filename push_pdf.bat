@echo off
echo This script will commit and push ONLY main.pdf
echo.

REM Navigate to the repository directory
cd /d "C:\Users\angel\OneDrive\Desktop\Project_Master"

REM Add only main.pdf, forcing if it's in .gitignore
echo Staging main.pdf...
git add -f "main.pdf"

REM Check if there are staged changes for main.pdf to avoid empty commits
git diff --staged --quiet --exit-code "main.pdf"
if %errorlevel% == 0 (
    echo No changes detected in main.pdf to commit.
    pause
    exit /b
)

REM Commit with a specific message for the PDF
set MSG="Update main.pdf (%date% %time%)"
echo Committing main.pdf...
git commit -m %MSG%

REM Push to the main branch
echo Pushing to remote...
git push origin main

echo.
echo Push of main.pdf complete.
pause
