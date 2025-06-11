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

- Download **Visual Studio's C++ build tools package**
  `https://visualstudio.microsoft.com/downloads/`, then install `Desktop development with C++`

- Clone this repository `git clone https://github.com/VIPTankz/Wii-RL.git`

---

### 2. Game ROM

Once you've installed the above, you will need to download the game, Mario Kart Wii.  
We cannot legally distribute this ROM, so you will need to acquire a Mario Kart Wii ROM yourself.
When you acquire the ROM, rename it to `mkw.iso` and put it in the directory `game`.

- **We use the European RMCP01 version of the game, `mkw.iso` (4.38GB).**  
  Please install this version to avoid other potential issues.
  (MD5 checksum e7b1ff1fabb0789482ce2cb0661d986e)

---

### 3. Setting Up This Repository

To correctly allow this repo to interact with Dolphin, please follow these steps:

1. Download **Felk's Fork of Dolphin**, which allows programmatic input to the emulator via Python. This can be done via running the script `download_dolphin.bat`.
   (The original repository can be found at `https://github.com/Felk/dolphin/releases`. However we highly recommend using our version listed above, as many settings have been changed, and we use a very specfic commit of this repository.)
2. Download the savestates (these control the AI's starting position) via running `download_savestates.bat`.

---

### 4. Installing Libraries

To install the relevant libraries, please do `pip install -r requirements.txt`.
(We do also include `environment.yml` if you want to install via conda).

---

### 5. Download a Pre-Trained Model

While we provide code to train your own models, we also include a model for you to run and test. To use this model, download the pytorch model file from `https://github.com/VIPTankz/Wii-RL/releases/tag/model`, and place this in this directory.

---
### 6. Running The AI with Dolphin

To first test whether everything is set up as intended, we recommend first running `BTR_test.py`. This will run the pretrained model installed in the last step, in two emulators in parallel.

To test if training on your machine works quickly, you can also run `BTR.py --testing 1`.

To actually do your own training, simply run `BTR.py`. This will use 4 instances of dolphin by default. This will put quite some strain on most PCs, so you may want to reduce this to 2 or 1 (You can also do 8 if you have a crazy good machine and don't mind your fans going crazy).

---

### 7. What to Expect

1. For the first 200k timesteps, the agent will simply execute a random policy, so don't expect to see any improvements during this time.
2. From 200k timesteps to 2M timesteps, the agent will slowly use fewer random actions, but during this period it may be hard to see any improvement.
3. At 2M-5M timesteps, you should be able to clearly see improvement. If not, something is messed up.
4. When testing this on an i9-13900k and RTX4090, it takes around 12 Hours to get an agent which can consistently finish the race.
5. When you start a training run, a new folder will be created (something like BTR_MarioKart2000M). This will contain a .png file with a graph of reward over time.
6. By default, the agent will run for 2 Billion frames (500 Million Timesteps). This is a LONG time, so don't be waiting around for it to finish.

Best of Luck!

