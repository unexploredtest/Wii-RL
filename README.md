# Wii-RL
AI-Tango's Official Wii Reinforcement Learning Repository
(https://www.youtube.com/@aitango)

Please watch the video which explains how to use this repository in detail! - (COMING SOON)

This repository contains code to allow Reinforcement Learning agents to to play Wii games. In this repo, we provide an example of:
- A Mario Kart Wii environment, using Luigi Circuit against Hard CPUs on 150cc.
- We provide a Reinforcement Learning algorithm (Beyond The Rainbow), which is setup and ready to interact with the environment.
- This algorithm is able to get first place in approximately one day of training using an RTX4090. This algorithm can still be run on lighter hardware, but may take slightly longer.

Beyond The Rainbow (BTR) algorithm, accepted at ICML 2025 (Poster) - https://arxiv.org/abs/2411.03820

Installation Instructions:
-Download Python 3.12 (It needs to be this version as the Dolphin Scripting Fork relies on this vesrion) - https://www.python.org/downloads/release/python-3120/
-Download Felk's Fork of Dolphin, which allows programmatic input to the emulator via python - https://github.com/Felk/dolphin/releases
-Download Visual Studio's C++ build tools package.

Once you've installed these, you will need to download the game, Mario Kart Wii. We cannot legally distribute this ROM, so you will need to acquire a Mario Kart Wii ROM.
We use the European RMCP01 version of the game, mkw.iso. Please install this version to avoid other potential issues.

To correctly allow this repo to interact with Dolphin, please follow these steps:
-After cloning this repo, please move your downloaded version of Dolphin into this folder. 
-Rename the dolphin folder, "dolphin0".
-Within the dolphin folder, locate the folder that contains the dolphin.exe. In this directory, create a blank file called portable.txt

Run Dolphin.exe, and follow these steps (many are optional, but it's what I use and has been tested):
-Options -> Configuration -> Paths -> Add the folder whever your Mario Kart .iso is saved.
-Graphics -> Enchancements -> Internal Resolution-> 640x528
-Graphics -> Backend -> Vulkan (optional, OpenGL works but slightly slower for me)
-Graphics -> Tick Auto-adjust Window Size
-Options -> Configuration -> Interface -> Disable (Confirm on Stop, Use Panic Handlers, Show on-screen display messages, Show active title). These are optional but I prefer them.
-Options -> Configuration -> General -> Speed Limit -> Unlimited (100% Works fine, but can run faster on unlimited. Also try not to kill your PC).
-Options -> Configuration -> Audio -> Audio Backend -> No Audio Output (If you forget to do this, RIP your ears).
-View -> Tick (Show Log, Show log configurations). This allows you to see debug messages.
