@echo off
set current=%~dp0
set root=%current%..
cd %root%
set pythonexe="c:\Python387_x64\python.exe"
if not exist %pythonexe% set pythonexe="c:\Python372_x64\python.exe"

@echo running 'python -m autopep8 --in-place --aggressive --aggressive -r'
%pythonexe% -m autopep8 --in-place --aggressive --aggressive -r onnxcustom tests examples setup.py doc/conf.py

@echo running 'python -m flake8 onnxcustom tests examples'
%pythonexe% -m flake8 onnxcustom tests examples setup.py doc/conf.py

if %errorlevel% neq 0 exit /b %errorlevel%
@echo Done Testing.
