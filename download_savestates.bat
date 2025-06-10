@echo off
setlocal

set "url=https://github.com/VIPTankz/Wii-RL/releases/download/savestates/MarioKartSaveStates.zip"
set "zipFile=savestates.zip"

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

echo Done!
endlocal
