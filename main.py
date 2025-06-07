import os
import pyautogui
import cv2
import numpy as np
import pytesseract
import itertools
import time
from typing import List, Tuple, Dict, Optional
import argparse # Import argparse

class WordGameBot:
    def __init__(self, debug_mode: bool = False): # Add debug_mode parameter
        self.letter_positions: List[Tuple[str, Tuple[int, int]]] = []
        self.available_letters: List[str] = []
        self.valid_words = set()
        self.screen_region: Optional[Tuple[int, int, int, int]] = None
        self.debug_mode = debug_mode # Store debug_mode
        self.load_word_dictionary()

        # Dynamically load all image files from a new folder for full-screen checks
        self.x_button_templates_full = self.load_templates_from_folder("collect_buttons") # New attribute

        self.level_button_template_path = "templates/level_button.png" # Path to your level button template
        self.lightning_strike_template_path = "templates/ligtning_strike.png"
        self.bluestacks_home_template_path = "templates/bluestacks_home_template.png"
        self.wordnut_game_template_path = "templates/wordnut_game_template.png"


        # Store both ROI configurations
        self.letter_rois_relative_7_hard = {
            "top_letter": (208, 95, 66, 66),
            "left_letter": (88, 150, 66, 66),
            "right_letter": (322, 150, 66, 66),
            "bottom_left_letter": (63, 277, 66, 66),
            "bottom_right_letter": (355, 277, 66, 66),
            "bottom_lower_left_letter": (140, 379, 66, 66),
            "bottom_lower_right_letter": (273, 379, 66, 66),
        }

        self.letter_rois_relative_7_letters = {
            "top_letter": (208, 78, 66, 66),
            "left_letter": (90, 132, 66, 66),
            "right_letter": (320, 132, 66, 66),
            "bottom_left_letter": (58, 260, 66, 66),
            "bottom_right_letter": (350, 260, 66, 66),
            "bottom_lower_left_letter": (138, 358, 66, 66),
            "bottom_lower_right_letter": (265, 358, 66, 66),
        }

        self.letter_rois_relative_6_letters = {
            "top_letter": (212, 98, 66, 66),
            "left_letter": (90, 171, 66, 66),
            "right_letter": (341, 171, 66, 66),
            "bottom_left_letter": (89, 315, 66, 66),
            "bottom_right_letter": (339, 313, 66, 66),
            "bottom_middle_letter": (210, 388, 66, 66),
        }

        self.letter_rois_relative_5_letters = {
            "top_letter": (210, 102, 66, 66),
            "left_letter": (78, 200, 66, 66),
            "right_letter": (347, 200, 66, 66),
            "bottom_left_letter": (127, 358, 66, 66),
            "bottom_right_letter": (295, 358, 66, 66),
        }

        # self.letter_rois_relative will be set dynamically by detect_game_layout
        self.letter_rois_relative = None

    # Add this new method to your WordGameBot class
    def find_and_click_button(self, button_template_path, label, dont_click=False) -> bool:
        """
        Attempts to find and click the 'Start Level' button using template matching.
        Returns True if clicked, False otherwise.
        """
        print("Attempting to find 'Start Level' button...")

        try:
            screenshot = pyautogui.screenshot()
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            screenshot_gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)

            template = cv2.imread(button_template_path)
            if template is None:
                print(f"Warning: Could not load {label} template: {button_template_path}. Skipping.")
                return False

            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            confidence_threshold = 0.7 # Adjust this confidence as needed for your button

            if max_val >= confidence_threshold:
                h, w = template_gray.shape
                center_x = max_loc[0] + w // 2
                center_y = max_loc[1] + h // 2

                print(f"{label} button found at ({center_x}, {center_y}) with confidence {max_val:.2f}")
                if not dont_click:
                    pyautogui.click(center_x, center_y)
                    print(f"Clicked {label} button. Waiting for game to load...")
                    time.sleep(3) # Give time for the level to load
                return True
            else:
                print(f"No {label} button found. Best confidence: {max_val:.2f}")
                return False

        except Exception as e:
            print(f"Error finding/clicking {label} button: {e}")
            return False

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

    def collect_buttons(self, full_screenshot_cv):
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

                confidence_threshold = 0.70 # Slightly lower confidence for full screen

                if max_val >= confidence_threshold:
                    h, w = template_gray.shape
                    center_x = max_loc[0] + w // 2
                    center_y = max_loc[1] + h // 2

                    print(f"X button found (FULL SCREEN) using template '{template_path}' at ({center_x}, {center_y}) with confidence {max_val:.2f}")
                    return (center_x, center_y)
            except Exception as e:
                print(f"Error during template matching for '{template_path}' in full screen: {e}. Skipping.")
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
            blurred_game_image = cv2.GaussianBlur(gray_game_image, (5, 5), 0)
            _, binary_game_image = cv2.threshold(blurred_game_image, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary_game_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            debug_image_contours = image.copy()
            cv2.drawContours(debug_image_contours, contours, -1, (255, 0, 255), 1)
            if self.debug_mode: cv2.imwrite("debug_05_all_contours_ocr.png", debug_image_contours) # Save this image

            # Main letter detection uses fixed ROIs
            for roi_name, (rel_x, rel_y, rel_w, rel_h) in self.letter_rois_relative.items():
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
                scale_factor = 6 # Increased scale factor for potentially better detail on thin characters like 'I'
                roi_height, roi_width = roi_gray.shape
                roi_resized = cv2.resize(roi_gray, (roi_width * scale_factor, roi_height * scale_factor),
                                        interpolation=cv2.INTER_CUBIC)

                # Optional: Adjust contrast/brightness before filtering
                # roi_resized = cv2.convertScaleAbs(roi_resized, alpha=1.3, beta=0) # Example adjustment

                roi_filtered = cv2.bilateralFilter(roi_resized, 5, 75, 75)

                best_char = None
                best_confidence = 0
                best_method_name = ""

                # Create a list of processed ROIs to try, prioritizing solid ones
                processed_roi_candidates = []

                # Increased kernel size for closing to ensure solidity
                kernel_solid_medium = np.ones((5,5), np.uint8) # Slightly larger kernel for solidification
                kernel_solid_small = np.ones((3,3), np.uint8)


                # Method 1: OTSU (often good for consistent lighting)
                _, roi_otsu_inv = cv2.threshold(roi_filtered, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                roi_otsu_inv_solid = cv2.morphologyEx(roi_otsu_inv, cv2.MORPH_CLOSE, kernel_solid_medium, iterations=1)
                processed_roi_candidates.append((roi_otsu_inv_solid, "otsu_inv_solid"))
                if self.debug_mode: cv2.imwrite(f"debug_roi_{roi_name}_otsu_inv_solid.png", roi_otsu_inv_solid)


                # Method 2: Simple Inverted Threshold (might work well if lighting is stable)
                _, roi_simple_inv = cv2.threshold(roi_filtered, 210, 190, cv2.THRESH_BINARY_INV) # Experiment with 150
                roi_simple_inv_solid = cv2.morphologyEx(roi_simple_inv, cv2.MORPH_CLOSE, kernel_solid_medium, iterations=1)
                processed_roi_candidates.append((roi_simple_inv_solid, "simple_inv_solid"))
                if self.debug_mode: cv2.imwrite(f"debug_roi_{roi_name}_simple_inv_solid.png", roi_simple_inv_solid)

                # Method 24: Simple Inverted Threshold (Works for F but not I)
                _, roi_simple_inv = cv2.threshold(roi_filtered, 190, 240, cv2.THRESH_BINARY_INV) # Experiment with 150
                roi_simple_inv_solid = cv2.morphologyEx(roi_simple_inv, cv2.MORPH_CLOSE, kernel_solid_medium, iterations=1)
                processed_roi_candidates.append((roi_simple_inv_solid, "simple_inv_solid"))
                if self.debug_mode: cv2.imwrite(f"debug_roi_{roi_name}_simple_inv_solid_2.png", roi_simple_inv_solid)

                # Loop through candidates, stopping if a high confidence match is found
                for img_variant, method_label in processed_roi_candidates:
                    if img_variant is None or img_variant.size == 0: continue

                    # Updated OCR configurations
                    ocr_configs = [
                        '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 13 --oem 3',
                        '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3', # Treat as a single word
                        '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6 --oem 3', # Treat as a single block
                        '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10 --oem 3',
                    ]

                    for config_idx, config in enumerate(ocr_configs):
                        try:
                            data = pytesseract.image_to_data(img_variant, config=config, output_type=pytesseract.Output.DICT)

                            for i in range(len(data['text'])):
                                text = data['text'][i].strip()
                                confidence = int(data['conf'][i])

                                if len(text) == 1 and text.isalpha() and text.isupper() and confidence > best_confidence:
                                    best_char = text
                                    best_confidence = confidence
                                    best_method_name = method_label + f" (config {config_idx})"
                                    # If you find a very high confidence match, you might want to break early
                                    if best_confidence > 90: # High confidence, likely correct
                                        break
                            if best_confidence > 90: # Break outer loop too
                                break
                        except Exception as e:
                            print(f"OCR error with method {method_label}, config {config_idx}: {e}")
                    if best_confidence > 90:
                        break # Break the loop over processed_roi_candidates if high confidence found

                # Accept result only if confidence is reasonable
                # Increased acceptance threshold
                if best_char and best_confidence > 60: # Adjusted threshold from 60 to 70
                    center_x_global = x_start + rel_w // 2 + self.screen_region[0]
                    center_y_global = y_start + rel_h // 2 + self.screen_region[1]
                    matched_letters_coords.append((best_char, (center_x_global, center_y_global)))
                    print(f"ACCEPTED OCR for '{roi_name}': {best_char} (confidence: {best_confidence}) - method: {best_method_name} at ({center_x_global}, {center_y_global})")
                else:
                    print(f"REJECTED OCR for '{roi_name}' - best result: '{best_char}' (confidence: {best_confidence})")

            if self.debug_mode: cv2.imwrite("debug_06_ocr_results.png", debug_image_ocr_results)

            return matched_letters_coords

    def find_circular_letters(self) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Captures the game area and then calls detect_letters_from_positions
        to find letters using the determined fixed ROIs.
        """
        if not self.screen_region:
            print("Error: Game area not defined. Cannot detect letters.")
            return []

        if not self.letter_rois_relative:
            print("Error: Letter ROIs not set. Cannot detect letters. Run detect_game_layout first.")
            return []

        image_cropped_to_region = self.capture_game_area()

        if image_cropped_to_region is None or image_cropped_to_region.size == 0:
            print("Error: No image captured for letter detection.")
            return []

        # The letter_rois_relative will already be set by detect_game_layout
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

    def detect_game_layout(self) -> bool:
        """
        Detects which game layout (5-letter or 6-letter) is present
        and sets self.screen_region and self.letter_rois_relative accordingly.
        It prioritizes the template with the highest match confidence.
        """
        print("Detecting game layout (5-letter or 6-letter)...")

        # Define templates to check and their corresponding ROI sets
        template_configs = [
            ("templates/game_area_template_7_letters.png", self.letter_rois_relative_7_letters, "7-letter"),
            ("templates/game_area_template_7_hard.png", self.letter_rois_relative_7_hard, "7-letter"),
            ("templates/game_area_template_6_letters.png", self.letter_rois_relative_6_letters, "6-letter"),
            ("templates/game_area_template_5_letters.png", self.letter_rois_relative_5_letters, "5-letter"),
        ]

        best_match_confidence = 0.0
        best_match_region = None
        best_match_rois = None
        best_match_layout_name = None

        # Capture screenshot once for all template matching attempts
        try:
            screenshot = pyautogui.screenshot()
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            screenshot_gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"Error capturing screenshot for layout detection: {e}")
            self.screen_region = None
            self.letter_rois_relative = None
            return False

        if screenshot_cv is None or screenshot_cv.size == 0:
            print("Failed to capture screenshot for layout detection.")
            self.screen_region = None
            self.letter_rois_relative = None
            return False

        for template_path, rois, layout_name in template_configs:
            try:
                template = cv2.imread(template_path)
                if template is None:
                    print(f"Warning: Could not load template image: {template_path}. Skipping.")
                    continue

                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

                result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                min_val, current_max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                print(f"Template '{layout_name}' matched with confidence: {current_max_val:.2f}")

                # Keep track of the best match found so far
                if current_max_val > best_match_confidence:
                    best_match_confidence = current_max_val
                    h, w = template_gray.shape
                    best_match_region = (max_loc[0], max_loc[1], w, h)
                    best_match_rois = rois
                    best_match_layout_name = layout_name

            except Exception as e:
                print(f"Error during template matching for '{template_path}': {e}. Skipping.")

        # Minimum acceptable confidence for ANY layout to be considered valid
        # This prevents selecting a low-confidence match if nothing truly good is found.
        minimum_acceptable_confidence = 0.75 # You can adjust this value

        if best_match_confidence >= minimum_acceptable_confidence:
            self.screen_region = best_match_region
            self.letter_rois_relative = best_match_rois
            print(f"Best game layout detected: {best_match_layout_name} at: {self.screen_region} with confidence: {best_match_confidence:.2f}")
            return True
        else:
            print(f"No game layout matched above the minimum acceptable confidence of {minimum_acceptable_confidence:.2f}. Best match was {best_match_layout_name} with {best_match_confidence:.2f}")
            self.screen_region = None
            self.letter_rois_relative = None
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
                time.sleep(0.5)
                # After spelling a word, re-check game area to see if level advanced or ad appeared
                # Call detect_game_layout to re-evaluate the screen state
                if not self.detect_game_layout(): # Use the new detection method
                    print("Game area disappeared or changed. Level might be complete or an ad is showing. Returning from play_level.")
                    return # Exit play_level, loop will handle ad/re-detection
            else:
                print(f"Failed to spell: {word} (likely missing letter positions or wrong order)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Word Game Bot for automating gameplay.")
    parser.add_argument('--debug', action='store_true',
                        help='If set, debug images will be saved for analysis.')
    args = parser.parse_args()

    bot = WordGameBot(debug_mode=args.debug) # Pass debug_mode to the bot

    while True:
        print("\n--- Starting New Cycle ---")

        # --- Stage 1: If game layout and ads are NOT detected, check for a 'Lightning Strike' button ---
        print("Checking for a 'Lightning Strike' button")
        lightning_strike_found = bot.find_and_click_button(bot.lightning_strike_template_path, 'Lightning Strike') # Use the new method
        if lightning_strike_found:
            # If button was found and clicked, invalidate game area to force re-detection on new level
            bot.screen_region = None
            bot.letter_rois_relative = None
            # The find_and_click_level_button already includes a sleep and prints for clicking
            continue # Go to the next cycle to detect game area/play

        # --- Stage 2: If game layout and ads are NOT detected, check for a 'Start Level' button ---
        print("Checking for a 'Start Level' button")
        start_button_found = bot.find_and_click_button(bot.level_button_template_path, 'Start Level') # Use the new method
        if start_button_found:
            # If button was found and clicked, invalidate game area to force re-detection on new level
            bot.screen_region = None
            bot.letter_rois_relative = None
            # The find_and_click_level_button already includes a sleep and prints for clicking
            continue # Go to the next cycle to detect game area/play

        # --- Stage 3: Detect the game layout ---
        # This will set bot.screen_region and bot.letter_rois_relative if successful
        if bot.detect_game_layout():
            print("Game layout successfully detected. Proceeding to play level.")
            bot.play_level()
            # After attempting to play, invalidate screen_region and ROIs
            # to force re-detection on the next cycle, as levels change.
            bot.screen_region = None
            bot.letter_rois_relative = None
            time.sleep(8) # Give some time for transition after a level or ad
            continue # Start next cycle to re-evaluate state
        
        # --- Stage 4: If game layout is NOT detected, check for exit/collect buttons ---
        print("Game layout not found. Checking for exit/collect Button...")
        full_screenshot = pyautogui.screenshot()
        full_screenshot_cv = cv2.cvtColor(np.array(full_screenshot), cv2.COLOR_RGB2BGR)
        exit_center = bot.collect_buttons(full_screenshot_cv)
        if exit_center:
            pyautogui.click(exit_center[0], exit_center[1])
            print("Button clicked. Waiting for game to load/resume...")
            time.sleep(5)
            bot.screen_region = None
            bot.letter_rois_relative = None
            continue # Re-evaluate state from the top

        # --- Stage 5: Assume ad is playing, restarting game ---
        time.sleep(1) # Add a small delay to prevent busy-waiting
        bot.find_and_click_button(bot.bluestacks_home_template_path, 'Bluestacks Home')
        # Invalidate game area to force re-detection on new level
        bot.screen_region = None
        bot.letter_rois_relative = None
        list_windows_found = bot.find_and_click_button(bot.wordnut_game_template_path, 'Word Nut')
        time.sleep(2) # Add a small delay to prevent busy-waiting
        continue # Go to the next cycle to detect game area/play
