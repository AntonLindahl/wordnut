import os
import pyautogui
import cv2
import numpy as np
import pytesseract
import itertools
import time
from typing import List, Tuple, Dict, Optional

# Set the path to the Tesseract executable if it's not in your system's PATH.
# Example for Windows:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # ADJUST THIS PATH
# Example for Linux/macOS:
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract' # Or wherever you installed it

class WordGameBot:
    def __init__(self):
        self.letter_positions: List[Tuple[str, Tuple[int, int]]] = []
        self.available_letters: List[str] = []
        self.valid_words = set()
        self.screen_region: Optional[Tuple[int, int, int, int]] = None
        self.load_word_dictionary()

        # Dynamically load all image files from the 'reklam' folder for top-screen checks
        self.x_button_templates_top = self.load_templates_from_folder("reklam") # Renamed for clarity
        # Dynamically load all image files from a new folder for full-screen checks
        self.x_button_templates_full = self.load_templates_from_folder("collect_buttons") # New attribute

        # *** PRECISELY CALCULATED letter_rois_relative BASED ON YOUR PROVIDED GAME AREA AND OCR DATA ***
        # Updated with the new values provided by the user.
        # Removed "center_letter" as requested.
        # Format: (x_relative, y_relative, width, height) relative to the top-left of the game area.

        self.letter_rois_relative = {
            "top_letter": (210, 100, 60, 60),
            "left_letter": (95, 173, 60, 60),
            "right_letter": (341, 173, 60, 60),
            "bottom_left_letter": (90, 317, 60, 60),
            "bottom_right_letter": (343, 317, 60, 60),
            "bottom_middle_letter": (210, 390, 60, 60),
        }

    def load_templates_from_folder(self, folder_path: str) -> List[str]:
        """
        Loads all image file paths from a given folder.
        Assumes image files have common extensions like .png, .jpg, .jpeg, .bmp.
        """
        template_paths = []
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                # Check if the file is an image (you can extend this list)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    template_paths.append(os.path.join(folder_path, filename))
            print(f"Loaded {len(template_paths)} templates from '{folder_path}'")
        else:
            print(f"Warning: Folder '{folder_path}' not found or is not a directory.")
        return template_paths

    def get_press_loc(self, max_loc, processed_template):
        h, w = processed_template.shape[:2]
        x, y = max_loc
        center = (x + w // 2, y + h // 2)
        return center

    def preprocess_for_template_matching(self, image: np.ndarray, debug_prefix: str = "", invert_output: bool = True, apply_morphology: bool = True) -> np.ndarray:
        """
        Preprocesses an image for contour detection and template matching.
        Saves intermediate steps for debugging.
        """
        if image is None or image.size == 0:
            print(f"Warning: preprocess_for_template_matching received an empty image for {debug_prefix}")
            return np.array([])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if debug_prefix: cv2.imwrite(f"{debug_prefix}_01_gray.png", gray)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        if debug_prefix: cv2.imwrite(f"{debug_prefix}_02_blurred.png", blurred)

        thresh_type = cv2.THRESH_BINARY_INV if invert_output else cv2.THRESH_BINARY

        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresh_type, 15, 5)
        if debug_prefix: cv2.imwrite(f"{debug_prefix}_03_thresh.png", thresh)

        processed = thresh

        if apply_morphology:
            kernel = np.ones((3, 3), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)

        if debug_prefix: cv2.imwrite(f"{debug_prefix}_04_processed.png", processed)

        return processed

    def exit_commersial(self):
        """
        Attempts to find and click the 'X' button using template matching.
        Tries multiple templates if provided.
        Checks only the top part of the screen first, then the full screen.
        """
        print("Attempting to find and click the 'X' button using template matching.")

        timeout = 30 # seconds to try to find the X button
        start_time = time.time()

        while time.time() - start_time < timeout:
            full_screenshot_cv = None
            try:
                full_screenshot = pyautogui.screenshot()
                full_screenshot_cv = cv2.cvtColor(np.array(full_screenshot), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Error capturing full screenshot for X button: {e}. Retrying...")
                time.sleep(1)
                continue

            if full_screenshot_cv is None or full_screenshot_cv.size == 0:
                print("Failed to capture full screenshot for X button detection. Retrying...")
                time.sleep(1)
                continue

            # --- Stage 1: Check the top part of the screen ---
            print("Checking the top part of the screen for X button.")
            top_region_height = int(full_screenshot_cv.shape[0] * 0.20) # Adjust percentage as needed
            top_screenshot_cv = full_screenshot_cv[0:top_region_height, :]
            top_screenshot_gray = cv2.cvtColor(top_screenshot_cv, cv2.COLOR_BGR2GRAY)

            # Create a debug image to show the scanned area for the top region
            #debug_scan_area_top = full_screenshot_cv.copy()
            #cv2.rectangle(debug_scan_area_top, (0, 0), (full_screenshot_cv.shape[1] - 1, top_region_height - 1), (0, 255, 0), 3)
            #cv2.putText(debug_scan_area_top, "Scan Area (Top Region)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #cv2.imwrite("debug_x_button_scan_area_top.png", debug_scan_area_top)
            #print(f"Saved 'debug_x_button_scan_area_top.png' showing the top scanned region.")

            for template_path in self.x_button_templates_top:
                try:
                    template = cv2.imread(template_path)
                    if template is None:
                        print(f"Warning: Could not load template image: {template_path}. Skipping.")
                        continue

                    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    result = cv2.matchTemplate(top_screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                    confidence_threshold = 0.90

                    if max_val >= confidence_threshold:
                        h, w = template_gray.shape
                        center_x = max_loc[0] + w // 2
                        center_y = max_loc[1] + h // 2 # Y is already global within the top region

                        print(f"X button found (TOP REGION) using template '{template_path}' at ({center_x}, {center_y}) with confidence {max_val:.2f}")
                        return (center_x, center_y)
                except Exception as e:
                    print(f"Error during template matching for '{template_path}' in top region: {e}. Skipping.")

            print("X button not detected in the top region. Moving to full screen check.")

            # --- Stage 2: Check the full screenshot ---
            full_screenshot_gray = cv2.cvtColor(full_screenshot_cv, cv2.COLOR_BGR2GRAY)

            for template_path in self.x_button_templates_full:
                try:
                    template = cv2.imread(template_path)
                    if template is None:
                        print(f"Warning: Could not load template image: {template_path}. Skipping.")
                        continue

                    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    result = cv2.matchTemplate(full_screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                    confidence_threshold = 0.80

                    if max_val >= confidence_threshold:
                        h, w = template_gray.shape
                        center_x = max_loc[0] + w // 2
                        center_y = max_loc[1] + h // 2

                        print(f"X button found (FULL SCREEN) using template '{template_path}' at ({center_x}, {center_y}) with confidence {max_val:.2f}")
                        return (center_x, center_y)
                except Exception as e:
                    print(f"Error during template matching for '{template_path}' in full screen: {e}. Skipping.")

            print("X button not detected using any template in full screen. Retrying cycle.")
            time.sleep(1)

        print(f"Timeout reached. X button not found after {timeout} seconds.")
        return False


    def capture_game_area(self) -> np.ndarray:
        if self.screen_region:
            screenshot = pyautogui.screenshot(region=self.screen_region)
        else:
            # Fallback to full screen if screen_region is not set (should be rare after setup)
            screenshot = pyautogui.screenshot()
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    def detect_letters_from_positions(self, image: np.ndarray) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Detects letters using pre-defined ROIs within the captured game area.
        Returns a list of (character, (x, y)) tuples for all detected letters.
        """
        matched_letters_coords: List[Tuple[str, Tuple[int, int]]] = []

        gray_game_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        debug_image_ocr_results = image.copy() # For drawing accepted OCR on

        # Temporary contour finding for debugging only (to generate debug_05_all_contours_ocr.png)
        # This part DOES NOT determine the letters for the bot's logic, only for visual aid.
        blurred_game_image = cv2.GaussianBlur(gray_game_image, (5, 5), 0)
        _, binary_game_image = cv2.threshold(blurred_game_image, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_game_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        debug_image_contours = image.copy()
        cv2.drawContours(debug_image_contours, contours, -1, (255, 0, 255), 1)
        cv2.imwrite("debug_05_all_contours_ocr.png", debug_image_contours) # Save this image

        # Main letter detection uses fixed ROIs
        for roi_name, (rel_x, rel_y, rel_w, rel_h) in self.letter_rois_relative.items():
            # Calculate absolute coordinates of the ROI within the current image (self.screen_region)
            x_start = rel_x
            y_start = rel_y
            x_end = rel_x + rel_w
            y_end = rel_y + rel_h

            if x_end <= x_start or y_end <= y_start or x_end > image.shape[1] or y_end > image.shape[0]:
                print(f"Skipping invalid ROI for '{roi_name}': out of bounds or zero dimensions.")
                continue

            roi_gray = gray_game_image[y_start:y_end, x_start:x_end]

            if roi_gray.size == 0 or roi_gray.shape[0] == 0 or roi_gray.shape[1] == 0:
                print(f"Skipping empty ROI for '{roi_name}' after extraction.")
                continue

            # --- Enhanced OCR Preprocessing ---
            # 1. Resize the ROI for better OCR accuracy
            scale_factor = 3  # Increase size by 3x
            roi_height, roi_width = roi_gray.shape
            roi_resized = cv2.resize(roi_gray, (roi_width * scale_factor, roi_height * scale_factor), 
                                    interpolation=cv2.INTER_CUBIC)

            # 2. Apply bilateral filter to reduce noise while preserving edges
            roi_filtered = cv2.bilateralFilter(roi_resized, 9, 75, 75)

            # Define thresholding methods to try
            threshold_methods = [
                ("simple", cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU), # Combined with OTSU
            ]

            best_char = None
            best_confidence = 0
            best_method_name = ""

            for method_name, thresh_type in threshold_methods:
                if method_name == "otsu" or method_name == "simple":
                    _, roi_thresh = cv2.threshold(roi_filtered, 0 if method_name == "otsu" else 127, 255, thresh_type)
                elif method_name == "adaptive":
                    roi_thresh = cv2.adaptiveThreshold(roi_filtered, 255, thresh_type, cv2.THRESH_BINARY_INV, 11, 2)
                
                # 4. Refined morphological operations
                kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

                # Remove small noise
                roi_cleaned = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)
                # Fill small gaps
                roi_cleaned = cv2.morphologyEx(roi_cleaned, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
                # Slightly thicken the letters
                roi_final = cv2.dilate(roi_cleaned, kernel_small, iterations=1)
                
                cv2.imwrite(f"debug_roi_{roi_name}_{method_name}.png", roi_final)

                # --- Enhanced OCR with Multiple Configurations ---
                ocr_configs = [
                    '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 13 --oem 3', # Good for single chars like 'I'
                    '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3',  # Treat the image as a single text line (more flexible than 13)
                ]

                # Try each OCR config for the current preprocessed image
                for config_idx, config in enumerate(ocr_configs):
                    try:
                        # Try normal image
                        data = pytesseract.image_to_data(roi_final, config=config, output_type=pytesseract.Output.DICT)
                        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                        texts = [text.strip() for text in data['text'] if text.strip()]
                        
                        if texts and confidences:
                            text = texts[0]
                            confidence = confidences[0]
                            if len(text) == 1 and text.isalpha() and text.isupper() and confidence > best_confidence:
                                best_char = text
                                best_confidence = confidence
                                best_method_name = method_name
                                # print(f"OCR candidate: {text} (confidence: {confidence}) - method: {method_name}, config: {config_idx}")
                        
                        # Try inverted image
                        roi_inverted = cv2.bitwise_not(roi_final)
                        data_inv = pytesseract.image_to_data(roi_inverted, config=config, output_type=pytesseract.Output.DICT)
                        confidences_inv = [int(conf) for conf in data_inv['conf'] if int(conf) > 0]
                        texts_inv = [text.strip() for text in data_inv['text'] if text.strip()]
                        
                        if texts_inv and confidences_inv:
                            text_inv = texts_inv[0]
                            confidence_inv = confidences_inv[0]
                            if len(text_inv) == 1 and text_inv.isalpha() and text_inv.isupper() and confidence_inv > best_confidence:
                                best_char = text_inv
                                best_confidence = confidence_inv
                                best_method_name = method_name + " (inverted)"
                                # print(f"OCR candidate (inverted): {text_inv} (confidence: {confidence_inv}) - method: {method_name}, config: {config_idx}")
                                
                    except Exception as e:
                        print(f"OCR error with method {method_name}, config {config_idx}: {e}")

            # Accept result only if confidence is reasonable
            if best_char and best_confidence > 30:  # Adjust threshold as needed
                center_x_global = x_start + rel_w // 2 + self.screen_region[0]
                center_y_global = y_start + rel_h // 2 + self.screen_region[1]
                matched_letters_coords.append((best_char, (center_x_global, center_y_global)))
                print(f"ACCEPTED OCR for '{roi_name}': {best_char} (confidence: {best_confidence}) - method: {best_method_name} at ({center_x_global}, {center_y_global})")
            else:
                print(f"REJECTED OCR for '{roi_name}' - best result: '{best_char}' (confidence: {best_confidence})")


        cv2.imwrite("debug_06_ocr_results.png", debug_image_ocr_results)

        return matched_letters_coords

    def find_circular_letters(self) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Captures the game area and then calls detect_letters_from_positions
        to find letters using fixed ROIs.
        """
        image_cropped_to_region = self.capture_game_area()

        if image_cropped_to_region is None or image_cropped_to_region.size == 0:
            print("Error: No image captured for letter detection.")
            return []

        # 6 letter game area
        self.letter_rois_relative = {
            "top_letter": (212, 98, 66, 60),
            "left_letter": (90, 171, 66, 60),
            "right_letter": (341, 171, 66, 60),
            "bottom_left_letter": (89, 315, 66, 60),
            "bottom_right_letter": (339, 315, 66, 60),
            "bottom_middle_letter": (210, 388, 66, 60),
        }
        detected_letters = self.detect_letters_from_positions(image_cropped_to_region)
        if len(detected_letters) <= 4:
            # Try 5 letter game area
            self.letter_rois_relative = {
                "top_letter": (210, 102, 66, 60),
                "left_letter": (78, 200, 66, 60),
                "right_letter": (347, 200, 66, 60),
                "bottom_left_letter": (127, 358, 66, 60),
                "bottom_right_letter": (295, 358, 66, 60),
                # "center_letter": (205, 222, 70, 70) # Removed as per user request
            }
            detected_letters = self.detect_letters_from_positions(image_cropped_to_region)

        self.letter_positions = detected_letters
        self.available_letters = [char for char, _ in detected_letters]

        return self.letter_positions

    def generate_all_words(self, letters: List[str]) -> List[str]:
        """Generate all possible word combinations from available letters"""
        words = set()
        for length in range(3, len(letters) + 1): #Only 3 letter and up words
            for perm in itertools.permutations(letters, length):
                words.add(''.join(perm))
        return list(words)

    def filter_valid_words(self, words: List[str]) -> List[str]:
        """Filter generated words against a dictionary of valid English words"""
        return [word for word in words if word.upper() in self.valid_words]

    def load_word_dictionary(self, dict_path: str = "words.txt"):
        """Load a dictionary of valid English words into self.valid_words"""
        try:
            with open(dict_path, 'r') as f:
                self.valid_words = set(word.strip().upper() for word in f.readlines())
            print(f"Loaded {len(self.valid_words)} words from dictionary.")
        except FileNotFoundError:
            print("Dictionary file not found. Please ensure 'words.txt' is in the same directory.")
            self.valid_words = set()

    def drag_word_path(self, word: str):
        """Drag across letters to spell a word"""
        path_coords = []
        temp_letter_positions_list = list(self.letter_positions)

        for char_in_word in word:
            found_pos = None
            for idx, (detected_char, pos) in enumerate(temp_letter_positions_list):
                if detected_char == char_in_word:
                    found_pos = pos
                    del temp_letter_positions_list[idx]
                    break
            if found_pos:
                path_coords.append(found_pos)
            else:
                print(f"Could not find an available position for letter '{char_in_word}' in word '{word}'.")
                return False

        if not path_coords:
            print(f"No path coordinates generated for word: {word}")
            return False

        start_pos = path_coords[0]
        pyautogui.moveTo(start_pos[0], start_pos[1])
        pyautogui.mouseDown()

        for pos in path_coords[1:]:
            pyautogui.moveTo(pos[0], pos[1], duration=0.1)
            time.sleep(0.05)

        pyautogui.mouseUp()
        time.sleep(0.5)
        return True

    def load_game_area_from_image(self, template_path: str = "templates/game_area_template_6_letters.png"):
        """Find the game area by matching it with a saved template image"""
        try:
            screenshot = pyautogui.screenshot()
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            template = cv2.imread(template_path)
            if template is None:
                print(f"Could not load template image: {template_path}")
                return False

            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            screenshot_gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > 0.8:
                h, w = template_gray.shape
                self.screen_region = (max_loc[0], max_loc[1], w, h)
                print(f"Game area found at: {self.screen_region}")
                print(f"Match confidence: {max_val:.2f}")
                return True
            else:
                print(f"Game area not found. Best match confidence: {max_val:.2f}")
                return False

        except Exception as e:
            print(f"Error loading game area from image: {e}")
            return False

    def setup_game_area(self, template_path: str = "templates/game_area_template_6_letters.png"):
        """Setup game area using template image instead of manual calibration"""
        print("Looking for game area using template image...")

        if self.load_game_area_from_image(template_path):
            print("Game area successfully detected!")
            return True
        else:
            print("Failed to detect game area. Ensure the game area template is correct and visible.")
            return False


    def play_level(self):
        """Main function to play a single level"""
        print("Starting level analysis...")

        self.letter_positions = self.find_circular_letters()

        if not self.available_letters:
            print("No letters detected! Check your setup, template images, or image processing parameters.")
            return

        print(f"Detected letters: {self.available_letters}")
        print(f"Letter positions: {self.letter_positions}")

        all_words = self.generate_all_words(self.available_letters)
        valid_words = self.filter_valid_words(all_words)

        print(f"Found {len(valid_words)} valid words to try: {valid_words}")

        valid_words.sort(key=len, reverse=True)

        for word in valid_words:
            print(f"Attempting word: {word}")
            success = self.drag_word_path(word)
            if success:
                time.sleep(1)
                # After spelling a word, re-check game area to see if level advanced or ad appeared
                if not self.load_game_area_from_image():
                    print("Game area disappeared. Level might be complete or an ad is showing. Returning from play_level.")
                    return # Exit play_level, loop will handle ad/re-detection
            else:
                print(f"Failed to spell: {word} (likely missing letter positions or wrong order)")

# Usage example
if __name__ == "__main__":
    bot = WordGameBot()

    # Initial setup: Find the game area once at the start.
    # If not found, it will try again in the loop.
    bot.setup_game_area()

    while True:
        print("\n--- Starting New Cycle ---")

        # --- Stage 1: Check for a "Start Level" Button ---
        start_button_found = False
        try:
            # You might need to adjust confidence based on your image
            # Replace 'level_button.png' with your actual start button image name
            level_button = pyautogui.locateOnScreen('templates/level_button.png', confidence=0.7) 
            if level_button:
                pyautogui.click(pyautogui.center(level_button))
                print("Clicked level start button. Waiting for game to load...")
                time.sleep(3) # Give time for the level to load
                bot.screen_region = None # Invalidate game area to force re-detection on new level
                start_button_found = True
                continue # Go to the next cycle to detect game area/play
            else:
                print("No specific level start button found (assuming game already in progress or next level loads automatically).")
        except pyautogui.PyAutoGUIException as e:
            print(f"PyAutoGUI error locating level button: {e}. Ensure the button image exists and is visible. Continuing...")
        
        if start_button_found:
            continue # If we clicked a start button, the state has changed, re-evaluate from the top.

        # Stage 1: If game area is known, attempt to play a level.
        if bot.screen_region:
            print("Game area is defined. Attempting to play level.")
            bot.play_level()
            # After playing a level, invalidate screen_region to force re-detection
            # in the next cycle. This accounts for level transitions or ads appearing.
            bot.screen_region = None
            time.sleep(2) # Give some time for transition or ad to appear
            continue # Start next cycle to re-evaluate state

        # Stage 2: If game area is NOT defined (or was invalidated), try to set it up.
        # This will also handle cases where an ad might be overlaying the game area.
        print("Game area not defined or invalidated. Attempting to set up game area.")
        if bot.setup_game_area():
            print("Game area successfully detected after re-attempt. Will try to play next cycle.")
            time.sleep(1) # Short delay before next cycle
            continue # Game area found, try playing in the next cycle

        # Stage 3: If game area still not found (even after re-attempt),
        # this might mean an ad is present, or the game is not active.
        # As a last resort, check for a commercial/exit button.
        print("\n--- Game area still not found. Checking for Commercial/Exit Button ---")
        exit_center = bot.exit_commersial()
        if exit_center:
            pyautogui.click(exit_center[0], exit_center[1])
            print("Commercial clicked. Waiting for game to load/resume...")
            time.sleep(5) # Give more time for the ad to close and game to appear
            bot.screen_region = None # Invalidate game area to force re-detection
        else:
            print("No commercial 'X' button found. Waiting before next cycle to avoid rapid clicking/detection failures...")
            time.sleep(2) # Short delay if nothing was found, to prevent busy-looping