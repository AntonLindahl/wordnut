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

        # Dynamically load all image files from the 'reklam' folder for top-screen checks
        self.x_button_templates_top = self.load_templates_from_folder("reklam") # Renamed for clarity
        # Dynamically load all image files from a new folder for full-screen checks
        self.x_button_templates_full = self.load_templates_from_folder("collect_buttons") # New attribute

        self.level_button_template_path = "templates/level_button.png" # Path to your level button template
        self.lightning_strike_template_path = "templates/ligtning_strike.png"
        self.google_play_template_path = "templates/google_play_template.png"
        self.list_windows_template_path = "templates/list_windows_template.png"
        self.word_nut_template_path = "templates/word_nut_template.png"


        # Store both ROI configurations
        self.letter_rois_relative_7_letters = {
            "top_letter": (208, 78, 66, 60),
            "left_letter": (90, 132, 66, 60),
            "right_letter": (320, 132, 66, 60),
            "bottom_left_letter": (58, 260, 66, 60),
            "bottom_right_letter": (355, 260, 66, 60),
            "bottom_lower_left_letter": (140, 360, 66, 60),
            "bottom_lower_right_letter": (265, 360, 66, 60),
        }

        self.letter_rois_relative_6_letters = {
            "top_letter": (212, 98, 66, 60),
            "left_letter": (90, 171, 66, 60),
            "right_letter": (341, 171, 66, 60),
            "bottom_left_letter": (89, 315, 66, 60),
            "bottom_right_letter": (339, 315, 66, 60),
            "bottom_middle_letter": (210, 388, 66, 60),
        }

        self.letter_rois_relative_5_letters = {
            "top_letter": (210, 102, 66, 60),
            "left_letter": (78, 200, 66, 60),
            "right_letter": (347, 200, 66, 60),
            "bottom_left_letter": (127, 358, 66, 60),
            "bottom_right_letter": (295, 358, 66, 60),
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
                    print("Clicked {label} button. Waiting for game to load...")
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

    def preprocess_for_template_matching(self, image: np.ndarray, debug_prefix: str = "", invert_output: bool = True, apply_morphology: bool = True) -> np.ndarray:
        """
        Preprocesses an image for contour detection and template matching.
        Saves intermediate steps for debugging if debug_mode is True.
        """
        if image is None or image.size == 0:
            print(f"Warning: preprocess_for_template_matching received an empty image for {debug_prefix}")
            return np.array([])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.debug_mode and debug_prefix: cv2.imwrite(f"{debug_prefix}_01_gray.png", gray)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.debug_mode and debug_prefix: cv2.imwrite(f"{debug_prefix}_02_blurred.png", blurred)

        thresh_type = cv2.THRESH_BINARY_INV if invert_output else cv2.THRESH_BINARY

        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresh_type, 15, 5)
        if self.debug_mode and debug_prefix: cv2.imwrite(f"{debug_prefix}_03_thresh.png", thresh)

        processed = thresh

        if apply_morphology:
            kernel = np.ones((3, 3), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)

        if self.debug_mode and debug_prefix: cv2.imwrite(f"{debug_prefix}_04_processed.png", processed)

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

            for template_path in self.x_button_templates_top:
                try:
                    template = cv2.imread(template_path)
                    if template is None:
                        print(f"Warning: Could not load template image: {template_path}. Skipping.")
                        continue

                    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    result = cv2.matchTemplate(top_screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                    confidence_threshold = 0.90 # High confidence for top region as X buttons are often clear

                    if max_val >= confidence_threshold:
                        h, w = template_gray.shape
                        center_x = max_loc[0] + w // 2
                        center_y = max_loc[1] + h // 2 # Y is already global within the top region

                        print(f"X button found (TOP REGION) using template '{template_path}' at ({center_x}, {center_y}) with confidence {max_val:.2f}")
                        return (center_x, center_y)
                except Exception as e:
                    print(f"Error during template matching for '{template_path}' in top region: {e}. Skipping.")

            print("X button not detected in the top region. Moving to full screen check.")

            self.collect_buttons(full_screenshot_cv)

            print("X button not detected using any template in full screen. Retrying cycle.")
            time.sleep(1)

        print(f"Timeout reached. X button not found after {timeout} seconds.")
        return False

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

                confidence_threshold = 0.80 # Slightly lower confidence for full screen

                if max_val >= confidence_threshold:
                    h, w = template_gray.shape
                    center_x = max_loc[0] + w // 2
                    center_y = max_loc[1] + h // 2

                    print(f"X button found (FULL SCREEN) using template '{template_path}' at ({center_x}, {center_y}) with confidence {max_val:.2f}")
                    return (center_x, center_y)
            except Exception as e:
                print(f"Error during template matching for '{template_path}' in full screen: {e}. Skipping.")

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
                scale_factor = 4 # Increased scale factor for potentially better detail on thin characters like 'I'
                roi_height, roi_width = roi_gray.shape
                roi_resized = cv2.resize(roi_gray, (roi_width * scale_factor, roi_height * scale_factor),
                                        interpolation=cv2.INTER_CUBIC)

                roi_filtered = cv2.bilateralFilter(roi_resized, 9, 75, 75)

                # Define thresholding methods to try
                # Added a specific simple threshold that might work well for clear 'I's
                threshold_methods = [
                    ("otsu_inv", cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU),
                    ("simple_inv", cv2.THRESH_BINARY_INV),
                    ("adaptive_mean_inv", "adaptive_mean_inv"),     # New
                    ("adaptive_gauss_inv", "adaptive_gauss_inv"),   # New
                    ("adaptive_mean", "adaptive_mean"),             # New
                    ("adaptive_gauss", "adaptive_gauss"),           # New
                ]

                best_char = None
                best_confidence = 0
                best_method_name = ""

                for method_name, thresh_type_flag in threshold_methods:
                    roi_thresh = None
                    if method_name == "otsu_inv":
                        # For OTSU, the second parameter (threshold value) is ignored.
                        _, roi_thresh = cv2.threshold(roi_filtered, 0, 255, thresh_type_flag)
                    elif method_name == "simple_inv":
                        # A fixed threshold (e.g., 150) can sometimes be robust for specific lighting
                        _, roi_thresh = cv2.threshold(roi_filtered, 150, 255, thresh_type_flag)
                    # Add more sophisticated adaptive thresholding if needed, e.g.,
                    elif method_name == "adaptive_gauss_inv":
                         roi_thresh = cv2.adaptiveThreshold(roi_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                            cv2.THRESH_BINARY_INV, 15, 5)


                    if roi_thresh is None: continue # Skip if thresholding failed

                    # Refined morphological operations
                    # Use slightly different kernels for better 'I' detection
                    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)) # Smaller vertical kernel for erosion
                    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)) # Smaller vertical kernel for dilation
                    
                    # Apply initial opening to remove small noise
                    roi_cleaned = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel_erode, iterations=1)
                    
                    # Dilate slightly to connect broken lines, but be careful not to merge 'I' with adjacent characters
                    roi_final = cv2.dilate(roi_cleaned, kernel_dilate, iterations=1)
                    
                    # Save processed ROI for debugging only if debug_mode is True
                    if self.debug_mode:
                        cv2.imwrite(f"debug_roi_{roi_name}_{method_name}.png", roi_final)
                        cv2.imwrite(f"debug_roi_{roi_name}_{method_name}_inverted.png", cv2.bitwise_not(roi_final)) # Save inverted too

                    # --- Enhanced OCR with Multiple Configurations ---
                    ocr_configs = [
                        '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 13 --oem 3', # Good for single chars like 'I'
                        '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10 --oem 3',  # Treat the image as a single text line
                        '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3',  # Treat the image as a single text line
                    ]

                    # Try each OCR config for the current preprocessed image and its inversion
                    for config_idx, config in enumerate(ocr_configs):
                        for img_variant, label_suffix in [(roi_final, ""), (cv2.bitwise_not(roi_final), " (inverted)")]:
                            try:
                                data = pytesseract.image_to_data(img_variant, config=config, output_type=pytesseract.Output.DICT)
                                
                                # It's better to iterate through all detected words, not just the first one
                                for i in range(len(data['text'])):
                                    text = data['text'][i].strip()
                                    confidence = int(data['conf'][i])

                                    if len(text) == 1 and text.isalpha() and text.isupper() and confidence > best_confidence:
                                        best_char = text
                                        best_confidence = confidence
                                        best_method_name = method_name + label_suffix + f" (config {config_idx})"
                                        # print(f"OCR candidate: {text} (confidence: {confidence}) - method: {best_method_name}")
                                        
                            except Exception as e:
                                print(f"OCR error with method {method_name}{label_suffix}, config {config_idx}: {e}")

                # Accept result only if confidence is reasonable
                if best_char and best_confidence > 30:  # Adjust threshold as needed
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
    parser.add_argument('--no-commercial', action='store_true',
                        help='If set, the bot will not attempt to exit commercials.')
    parser.add_argument('--debug', action='store_true',
                        help='If set, debug images will be saved for analysis.')
    args = parser.parse_args()

    bot = WordGameBot(debug_mode=args.debug) # Pass debug_mode to the bot

    while True:
        print("\n--- Starting New Cycle ---")

        # --- Stage 3: If game layout and ads are NOT detected, check for a 'Lightning Strike' button ---
        print("No game detected and no ad button found. Checking for a 'Lightning Strike' button or waiting.")
        lightning_strike_found = bot.find_and_click_button(bot.lightning_strike_template_path, 'Lightning Strike') # Use the new method
        if lightning_strike_found:
            # If button was found and clicked, invalidate game area to force re-detection on new level
            bot.screen_region = None
            bot.letter_rois_relative = None
            # The find_and_click_level_button already includes a sleep and prints for clicking
            continue # Go to the next cycle to detect game area/play

        # --- Stage 3: If game layout and ads are NOT detected, check for a 'Start Level' button ---
        print("No game detected and no ad button found. Checking for a 'Start Level' button or waiting.")
        start_button_found = bot.find_and_click_button(bot.level_button_template_path, 'Start Level') # Use the new method
        if start_button_found:
            # If button was found and clicked, invalidate game area to force re-detection on new level
            bot.screen_region = None
            bot.letter_rois_relative = None
            # The find_and_click_level_button already includes a sleep and prints for clicking
            continue # Go to the next cycle to detect game area/play

        # --- Stage 1: Detect the game layout (5-letter or 6-letter) first ---
        # This will set bot.screen_region and bot.letter_rois_relative if successful
        if bot.detect_game_layout():
            print("Game layout successfully detected. Proceeding to play level.")
            bot.play_level()
            # After attempting to play, invalidate screen_region and ROIs
            # to force re-detection on the next cycle, as levels change.
            bot.screen_region = None
            bot.letter_rois_relative = None
            time.sleep(2) # Give some time for transition after a level or ad
            continue # Start next cycle to re-evaluate state
        
        if args.no_commercial: # Use args.no_commercial directly
            full_screenshot = pyautogui.screenshot()
            full_screenshot_cv = cv2.cvtColor(np.array(full_screenshot), cv2.COLOR_RGB2BGR)
            bot.collect_buttons(full_screenshot_cv)
        else:
            # --- Stage 2: If game layout is NOT detected, check for commercials/exit buttons ---
            print("Game layout not found. Checking for Commercial/Exit Button...")
            exit_center = bot.exit_commersial()
            if exit_center:
                pyautogui.click(exit_center[0], exit_center[1])
                print("Commercial clicked. Waiting for game to load/resume...")
                time.sleep(5) # Give more time for the ad to close and game to appear
                # After an ad, the game state might have changed, so invalidate current detections
                bot.screen_region = None
                bot.letter_rois_relative = None
                continue # Re-evaluate state from the top

        # --- Stage 3: If game layout and ads are NOT detected, check for other persistent buttons ---
        print("Checking for failed commercial press or other persistent buttons...")
        google_play_found = bot.find_and_click_button(bot.google_play_template_path, 'Google Play', dont_click=True)
        if google_play_found:
            print("Google Play button detected. Attempting to navigate back to Word Nut.")
            # Invalidate game area to force re-detection on new level
            bot.screen_region = None
            bot.letter_rois_relative = None
            list_windows_found = bot.find_and_click_button(bot.list_windows_template_path, 'List Windows')
            # If the list windows button was found and clicked, try to find and click Word Nut
            if list_windows_found:
                word_nut_found = bot.find_and_click_button(bot.word_nut_template_path, 'Word Nut')
            continue # Go to the next cycle to detect game area/play

        print("No specific action taken. Waiting before next cycle...")
        time.sleep(5) # Add a small delay to prevent busy-waiting