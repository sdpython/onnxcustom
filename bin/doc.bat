@echo off
set current=%~dp0
set root=%current%..
cd %root%
set pythonexe="c:\Python372_x64\python.exe"
if not exist %pythonexe% set pythonexe="c:\Python370_x64\python.exe"

@echo running 'python -m sphinx -T -b html doc dist/html'
%pythonexe% -m sphinx -T -b html doc dist/html

if %errorlevel% neq 0 exit /b %errorlevel%
@echo Done Testing.
