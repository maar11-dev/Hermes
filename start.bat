@echo off
chcp 65001 >nul
:: Hermes - Script de arranque (Windows)

echo.
echo   =========================================
echo     Hermes - Agente RAG
echo   =========================================
echo.

SET SCRIPT_DIR=%~dp0
SET BACKEND_DIR=%SCRIPT_DIR%backend
SET VENV_DIR=%SCRIPT_DIR%.venv
SET PYTHON_CMD=

:: Crear .env si no existe
IF NOT EXIST "%SCRIPT_DIR%.env" (
  COPY "%SCRIPT_DIR%.env.example" "%SCRIPT_DIR%.env" >nul
  echo   Creado .env desde .env.example
)
COPY "%SCRIPT_DIR%.env" "%BACKEND_DIR%.env" >nul 2>&1

:: Detectar Python (preferir launcher py)
where py >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
  SET PYTHON_CMD=py -3
) ELSE (
  where python >nul 2>&1
  IF %ERRORLEVEL% EQU 0 SET PYTHON_CMD=python
)

IF "%PYTHON_CMD%"=="" (
  echo   ERROR: No se encontro Python.
  echo   Instala Python 3.10+ desde https://python.org
  echo   Activa la opcion "Add Python to PATH" durante la instalacion.
  pause
  EXIT /B 1
)

:: Crear entorno virtual
IF NOT EXIST "%VENV_DIR%\Scripts\python.exe" (
  IF EXIST "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
  echo   Creando entorno virtual Python...
  %PYTHON_CMD% -m venv "%VENV_DIR%"
)

SET VENV_PYTHON=%VENV_DIR%\Scripts\python.exe

IF NOT EXIST "%VENV_PYTHON%" (
  echo   ERROR: No se pudo crear el entorno virtual.
  pause
  EXIT /B 1
)

echo   Instalando dependencias...
"%VENV_PYTHON%" -m pip install -q --upgrade pip
"%VENV_PYTHON%" -m pip install -q -r "%BACKEND_DIR%\requirements.txt"

mkdir "%SCRIPT_DIR%data" 2>nul

:: Leer LLM_PROVIDER del .env
SET LLM_PROVIDER=ollama
FOR /F "tokens=2 delims==" %%A IN ('findstr /B "LLM_PROVIDER" "%SCRIPT_DIR%.env" 2^>nul') DO SET LLM_PROVIDER=%%A
SET LLM_PROVIDER=%LLM_PROVIDER: =%

IF /I NOT "%LLM_PROVIDER%"=="ollama" GOTO :START_SERVER

:: Comprobar si ollama esta instalado
where ollama >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
  echo.
  echo   AVISO: Ollama no encontrado en el sistema.
  echo   Instala Ollama desde: https://ollama.com/download
  echo   O edita .env y pon LLM_PROVIDER=anthropic
  echo.
  pause
  EXIT /B 1
)

:: Leer modelo del .env
SET OLLAMA_MODEL=llama3.2
FOR /F "tokens=2 delims==" %%A IN ('findstr /B "OLLAMA_MODEL" "%SCRIPT_DIR%.env" 2^>nul') DO SET OLLAMA_MODEL=%%A
SET OLLAMA_MODEL=%OLLAMA_MODEL: =%

:: Comprobar si el modelo ya esta descargado
ollama list 2>nul | findstr /I "%OLLAMA_MODEL%" >nul
IF %ERRORLEVEL% NEQ 0 (
  echo   Descargando modelo %OLLAMA_MODEL% (solo la primera vez, puede tardar...^)
  ollama pull %OLLAMA_MODEL%
  IF %ERRORLEVEL% NEQ 0 (
    echo   ERROR: No se pudo descargar el modelo. Comprueba tu conexion.
    pause
    EXIT /B 1
  )
) ELSE (
  echo   Modelo %OLLAMA_MODEL% ya disponible.
)

:: Arrancar ollama serve si no esta corriendo
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /I "ollama.exe" >nul
IF %ERRORLEVEL% NEQ 0 (
  echo   Arrancando servidor Ollama en segundo plano...
  start /B "" ollama serve >nul 2>&1
  timeout /T 2 /NOBREAK >nul
) ELSE (
  echo   Servidor Ollama ya en ejecucion.
)

:START_SERVER
echo.
echo   Servidor iniciado en http://localhost:8000
echo   Abre esa URL en tu navegador
echo   Ctrl+C para detener
echo.

CD /D "%BACKEND_DIR%"
"%VENV_PYTHON%" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
