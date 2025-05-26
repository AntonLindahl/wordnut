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
   - Switch BlueStacks to **fullscreen mode** for optimal script performance.

4. **Run the Script**  
   - Clone this repository.
   - Setup virtual env (python -m venv venv)
   - Activate virtual env (for windows powershell this could be needed `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then `.\venv\Scripts\Activate.ps1`) 
   - Install requirements (pip install -r requirements.txt)
   - Run program (python .\main.py)
---

## Potential issues

1. **Resolution**
    Not sure about the game resolution on different computers. There are two definitions for letter_rois_relative where the letters get a box to find them easier. If letters are incorrect
    you can check the debug_roi*.png files. These should contain perfekt letters, if not modify the values until the letters are in the pictures.

2. **Start buttons**
    Some popups are made in the begining, these are not handled. Either manually fix some levels or screenshot the button and place in collect_buttons folder.

3. **Commercial**
    Not all exit commercial picture are added, or they might be changed. Take a screenshot of the button that should exit the commercial, add it in reklam folder and re-run the script.

4. **Missing words**
    There are not that many good word lists. If a word is missing please add it to words.txt and.

## Improvements

1. It should be possible to use the different game_area_templates to get correct letter amount instead of dumb letter_rois_relative test. 
2. Some commercial are pressed and play store is opened. It should be possible to do a function to return to the game. I have only seen commercials fail once and then work.
3. Increase speed with less sleep or some advanced usage of the result crossword play area.
