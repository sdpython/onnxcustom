@echo off
set current=%~dp0
set root=%current%..
cd %root%
set pythonexe="python.exe"

@echo running 'python -m unittest discover tests'
%pythonexe% -m unittest discover tests --verbose

if %errorlevel% neq 0 exit /b %errorlevel%
@echo Done Testing.