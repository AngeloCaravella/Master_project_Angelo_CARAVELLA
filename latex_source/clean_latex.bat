@echo off
echo Cleaning LaTeX auxiliary files...

REM Delete auxiliary files from the parent directory (Project_Master)
del /f /q "..\*.aux" "..\*.bbl" "..\*.bcf" "..\*.blg" "..\*.lof" "..\*.lot" "..\*.out" "..\*.toc" "..\*.run.xml"

REM Delete auxiliary files from the current directory (latex_source)
del /f /q "*.aux" "*.bbl" "*.bcf" "*.blg" "*.lof" "*.lot" "*.out" "*.toc" "*.run.xml" "*.synctex(busy)"

echo.
echo Cleanup complete.
echo You can now run compile_pdf.bat to recompile your thesis.
pause
