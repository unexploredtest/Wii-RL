# AI-Tango's Official Wii Reinforcement Learning Repository

[https://www.youtube.com/@aitango](https://www.youtube.com/@aitango)

---

**Please watch the video which explains how to use this repository in detail!**  
(COMING SOON)

---

## Overview

This repository contains code to allow Reinforcement Learning agents to play Wii games.  
In this repo, we provide an example of:

- A Mario Kart Wii environment, using Luigi Circuit against Hard CPUs on 150cc.
- A Reinforcement Learning algorithm (Beyond The Rainbow), which is setup and ready to interact with the environment.

This algorithm is able to get first place in approximately one day of training using an RTX4090.  
The algorithm can still be run on lighter hardware, but may take slightly longer.

**Beyond The Rainbow (BTR) algorithm, accepted at ICML 2025 (Poster):**  
https://arxiv.org/abs/2411.03820

---

## Installation Instructions

### 1. Prerequisites

- Download **Python 3.12**  
  _(It needs to be this version as the Dolphin Scripting Fork relies on it)_  
  https://www.python.org/downloads/release/python-3120/

- Download **Felk's Fork of Dolphin**, which allows programmatic input to the emulator via Python  
  https://github.com/Felk/dolphin/releases

- Download **Visual Studio's C++ build tools package**

---

### 2. Game ROM

Once you've installed the above, you will need to download the game, Mario Kart Wii.  
We cannot legally distribute this ROM, so you will need to acquire a Mario Kart Wii ROM yourself.

- **We use the European RMCP01 version of the game, `mkw.iso` (4.38GB).**  
  Please install this version to avoid other potential issues.

---

### 3. Setting Up This Repository

To correctly allow this repo to interact with Dolphin, please follow these steps:

1. After cloning this repo, move your downloaded version of Dolphin into this folder.
2. Rename the Dolphin folder to `dolphin0`.
3. Within the `dolphin0` folder, locate the directory that contains `dolphin.exe`.
4. In this directory, create a blank file called `portable.txt`.

---

### 4. Configuring Dolphin

Run `Dolphin.exe` and follow these steps (many are optional, but these settings have been tested):

- Options → Configuration → Paths → Add the folder where your Mario Kart `.iso` is saved.
- Graphics → Enhancements → Internal Resolution → 640x528
- Graphics → Backend → Vulkan (optional, OpenGL works but is slightly slower for me)
- Graphics → Tick **Auto-adjust Window Size**
- Options → Configuration → Interface → Disable:
    - Confirm on Stop
    - Use Panic Handlers
    - Show on-screen display messages
    - Show active title  
  (These are optional but are my preference.)
- Options → Configuration → General → Speed Limit → Unlimited  
  (100% works fine, but can run faster on unlimited. Also, try not to overload your PC.)
- Options → Configuration → Audio → Audio Backend → No Audio Output  
  (If you forget to do this, the audio can be very loud.)
- View → Tick **Show Log** and **Show log configurations**.  
  (This allows you to see debug messages.)

---

### 5. Installing Libraries

To install the relevant libraries, please do `pip install requirements.txt`.

---

### 6. Loading Savestates

This script requires some savestates to load from (ie the start of the race).
In this repository, we include some different savestates which you will need.
Take these savestates, and drag them into the following folder:

`dolphin0\User\StatesSaves`

---
### 7. Running The AI with Dolphin

You can control how many instances of Dolphin to run in parallel. On high Spec machines with many cores, I recommend 4 (or 8 if you're cooling system is really good).
Be warned however, this will put some serious strain on your machine.

However many instances of Dolphin you want to run, you will need that many installations of Dolphin. 
For 4 instances, you will need to copy your `dolphin0` folder, such that you have 4 folders named `dolphin0`, `dolphin1`, `dolphin2`, `dolphin3`.

To test your installiation, I recommend first running `python DolphinEnv.py`. This will allow you to control the race yourself and test things are working.

Then, please test the AI works by running `python BTR.py --testing 1`. 

Once you've confirmed things work, run `python BTR.py`, which will run the Beyond The Rainbow algorithm on the Mario Kart environment.

---

### 8. What to Expect

1. For the first 200k timesteps, the agent will simply execute a random policy, so don't expect to see any improvements during this time.
2. From 200k timesteps to 2M timesteps, the agent will slowly use fewer random actions, but during this period it may be hard to see any improvement.
3. At 2M-5M timesteps, you should be able to clearly see improvement. If not, something is messed up.
4. When testing this on an i9-13900k and RTX4090, it takes around 12 Hours to get an agent which can consistently finish the race.
5. When you start a training run, a new folder will be created (something like BTR_MarioKart2000M). This will contain a .png file with a graph of reward over time.
6. By default, the agent will run for 2 Billion frames (500 Million Timesteps). This is a LONG time, so don't be waiting around for it to finish.

Best of Luck!

