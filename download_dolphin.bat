@echo off
setlocal

set "url=https://github.com/VIPTankz/Wii-RL/releases/download/dolphin/dolphin0.zip"
set "outputDir=dolphin0\"
set "zipFile=Dolphin.zip"
set "decompressedDir=%outputDir%"

echo Downloading ZIP file...
curl -L -o "%zipFile%" "%url%"
if errorlevel 1 (
    echo Failed to download the file.
    exit /b 1
)

tar -xf "%zipFile%"

if errorlevel 1 (
    echo Extraction failed.
    exit /b 1
)

echo Files extracted to "%outputDir%".

@REM :: Clean up the downloaded ZIP file
del "%zipFile%"

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
