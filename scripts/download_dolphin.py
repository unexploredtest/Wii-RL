import os
from pathlib import Path
import platform

from common import download_file, extract_zip

DOLPHIN_WIN_URL = "https://github.com/VIPTankz/Wii-RL/releases/download/dolphin/dolphin0.zip"
DOLPHIN_MAC_ARM_URL = "https://github.com/unexploredtest/dolphin/releases/download/dolphin-wii-rl/DolphinMacArm.zip"
DOLPHIN_MAC_X86_URL = "https://github.com/unexploredtest/dolphin/releases/download/dolphin-wii-rl/DolphinMacx86.zip"
ZIP_NAME = "Dolphin.zip"

def main():
    current_directory = Path.cwd()

    if(platform.system() == "Windows"):
        download_file(DOLPHIN_WIN_URL, ZIP_NAME)
    elif(platform.system() == "Darwin" and platform.machine() == "arm64"):
        download_file(DOLPHIN_MAC_ARM_URL, ZIP_NAME)
    elif(platform.system() == "Darwin" and platform.machine() == "x86_64"):
        download_file(DOLPHIN_MAC_X86_URL, ZIP_NAME)
    else:
        raise RuntimeError(f"The operating system '{platform.system()}' is not supported.")
    
    extract_zip(ZIP_NAME, current_directory)

    os.remove(ZIP_NAME)

    print("Done!")


if __name__ == "__main__":
    main()