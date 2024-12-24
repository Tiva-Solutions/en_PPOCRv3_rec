from collections import deque, Counter
import cv2
from datetime import datetime
import re
import os
import numpy as np

def ocr_text(image, ocr_model):
    """
    Perform OCR on the cropped license plate image to extract text.
    :param image: The cropped image of the license plate.
    :param ocr_model: The OCR model (PaddleOCR).
    :return: Extracted text from the image or None if OCR fails.
    """
    PATTERN = r"\b\d{1,2}(?:[A-Z]{1,2}|[A-Z]\d)\s?\d{4,5}\b"
    try:
        results = ocr_model.ocr(image, det=True, rec=True, cls=False)
        if results:
            text = " ".join([line[1][0] for line in results[0]])
            #text = re.sub(r"[.\-]", "", text)  # Clean text
            text = re.sub(r'[^A-Za-z0-9]', '', text)
            matches = re.findall(PATTERN, text)
            return matches[0] if matches else None
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return None
    
class FrameInfo:
    def __init__(self, conf, cropped_img, timestamp):
        self.conf = conf
        self.cropped_img = cropped_img
        self.timestamp = timestamp

class TrackInfo:
    def __init__(self):
        self.best_frames = deque(maxlen=5)
        self.text = None
        self.active = True
        self.final_img = None
        self.final_timestamp = None
        self.timestamp_str = None
        self.output_path = None

    def add_frame(self, frame_info):
        if len(self.best_frames) < 5:
            self.best_frames.append(frame_info)
            self.best_frames = deque(sorted(self.best_frames, 
                                          key=lambda x: x.conf, 
                                          reverse=True), 
                                   maxlen=5)
        else:
            if frame_info.conf > self.best_frames[-1].conf:
                self.best_frames.pop()
                self.best_frames.append(frame_info)
                self.best_frames = deque(sorted(self.best_frames, 
                                              key=lambda x: x.conf, 
                                              reverse=True), 
                                       maxlen=5)

    def process_inactive(self, ocr_model, output_folder, detected_plates_cache):
        # If already active or text is already in detected plates cache, return
            if self.active or (self.text and self.text in detected_plates_cache):
                return

            if not self.active and self.text is None:
                texts = []
                for frame_info in self.best_frames:
                    plate_text = ocr_text(frame_info.cropped_img, ocr_model)
                    if plate_text:
                        ## Add color code deteted
                        # color_code = detect_plate_color(frame_info.cropped_img)
                        # if color_code:
                        #     plate_text += color_code
                        texts.append(plate_text)
                
                if texts:
                    text_counts = Counter(texts)
                    most_common_text = text_counts.most_common(1)[0][0]
                    
                    # Check if the most common text is already in detected plates cache
                    if most_common_text in detected_plates_cache:
                        return
                    
                    self.text = most_common_text
                    
                    best_conf = -1
                    for i, frame_info in enumerate(self.best_frames):
                        if i < len(texts) and texts[i] == most_common_text:
                            if frame_info.conf > best_conf:
                                best_conf = frame_info.conf
                                self.final_img = frame_info.cropped_img
                                self.final_timestamp = frame_info.timestamp
                    
                    if self.final_img is not None:
                        clean_text = most_common_text.replace(" ", "_")
                        self.timestamp_str = self.final_timestamp.strftime("%Y%m%d_%H%M%S")
                        self.output_path = os.path.join(output_folder, "image", f"{clean_text}_{self.timestamp_str}.jpg")
                        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                        cv2.imwrite(self.output_path, self.final_img)

                        # Lưu thông tin vào file labels.txt
                        label_file_path = os.path.join(output_folder, "labels.txt")
                        with open(label_file_path, "a") as label_file:
                            label_file.write(f"{self.output_path}\t{clean_text}\n")