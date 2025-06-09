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

- **We use the European RMCP01 version of the game, `mkw.iso`.**  
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
