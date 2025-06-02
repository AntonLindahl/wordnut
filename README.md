# Automation of Word Nut Game

Automate the gameplay of the Word Nut mobile game using this script. This project is designed to simplify repetitive gameplay by running the game on a virtual Android device in fullscreen mode.

## Installation Guide

Follow these steps to get started:

1. **Sign up on Freecash**  
   Earn rewards while using your system for automation tasks.  
   👉 [Join Freecash using my referral link](https://freecash.com/r/FreeRiches)

2. **Download BlueStacks**  
   BlueStacks is an Android emulator for running mobile apps on your PC.
   🔗 [Download BlueStacks](https://www.bluestacks.com/)

   Download freecash in bluestacks and take the wordnut offer. This should link you to the game and you should allow being tracked.

3. **Set Up Your Environment**  
   - Install Python
   - Install Tesseract [Windows install link](https://github.com/UB-Mannheim/tesseract/wiki)
   - Download newer eng.traineedata [Github download link](https://github.com/tesseract-ocr/tessdata/raw/refs/heads/main/eng.traineddata) and place in tessdata folder (should be C:\Program Files\Tesseract-OCR\tessdata)
   - Open Word Nut inside BlueStacks. Make sure freecash has registered the install game reward.
   - Switch BlueStacks to **fullscreen mode** for script to work.

4. **Run the Script**  
   - Clone this repository.
   - Setup virtual env (python -m venv venv)
   - Activate virtual env (for windows powershell this could be needed `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then `.\venv\Scripts\Activate.ps1`) 
   - Install requirements (pip install -r requirements.txt)
   - Run program (python .\main.py). If you want to buy the no commercials the command `python .\main.py --no-commercial` can be used
---

## Disclamer

1. **First levels**
    The first levels are not automated. These needs to be played until 5 letter game areas are used. This is level 18 i think.

2. **Commercial**
    Not all exit commercial picture are added, or they might be changed. Take a screenshot of the button that should exit the commercial, add it in reklam folder and re-run the script.
