@echo off
setlocal

set "url=https://github.com/unexploredtest/dolphin/releases/download/dolphin-mariokart/Dolphin.zip"
set "outputDir=dolphin0\"
set "zipFile=%outputDir%\Dolphin.zip"
set "decompressedDir=%outputDir%"

:: Create output directory if it doesn't exist
if not exist "%outputDir%" (
    mkdir "%outputDir%"
)

echo Downloading ZIP file...
curl -L -o "%zipFile%" "%url%"
if errorlevel 1 (
    echo Failed to download the file.
    exit /b 1
)

tar -xf "%zipFile%" -C %outputDir%

if errorlevel 1 (
    echo Extraction failed.
    exit /b 1
)

echo Files extracted to "%outputDir%".

@REM @REM :: Clean up the downloaded ZIP file
@REM del "%zipFile%"

@REM set "filename=portable.txt"  :: Replace with your desired file name

@REM @REM :: Create an empty file
@REM type nul > "%outputDir%\%filename%"

@REM echo File "%outputDir%\%filename%" created.

echo Creating Dolphin 1...
xcopy "%outputDir%" "dolphin1" /E /I /Y
if errorlevel 1 (
    echo Failed to copy the directory.
    exit /b 1
)

echo Creating Dolphin 2...
xcopy "%outputDir%" "dolphin2" /E /I /Y
if errorlevel 1 (
    echo Failed to copy the directory.
    exit /b 1
)

echo Creating Dolphin 3...
xcopy "%outputDir%" "dolphin3" /E /I /Y
if errorlevel 1 (
    echo Failed to copy the directory.
    exit /b 1
)

echo Done!
endlocal
