@echo off
REM ====================================================
REM Batch file to compile LaTeX with Biber automatically
REM This script assumes it is run from within the same
REM directory as the main.tex file.
REM The final PDF and auxiliary files will be placed
REM in the parent directory.
REM ====================================================

REM Step 1: Compile LaTeX to generate .bcf in parent dir
pdflatex -interaction=nonstopmode -output-directory=.. main.tex

REM Step 2: Run Biber (specify input and output dir)
biber --input-directory=. --output-directory=.. main

REM Step 3: Compile LaTeX again to include bibliography
pdflatex -interaction=nonstopmode -output-directory=.. main.tex

REM Step 4: Compile LaTeX one more time for cross-references
pdflatex -interaction=nonstopmode -output-directory=.. main.tex

REM Done
echo Compilation finished! PDF generated in parent directory.
pause
