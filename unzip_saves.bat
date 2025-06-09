@echo off
setlocal

set "inputDir=savestates"
set "outputDir=savestates"
set "baseName=RMCP01"

for %%f in ("%inputDir%\%baseName%*.zip") do (
    echo Extracting "%%f" to "%outputDir%"
    tar -xf "%%f" -C "%outputDir%"
)

endlocal