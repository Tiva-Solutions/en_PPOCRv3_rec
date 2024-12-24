import cv2
import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os
import threading
import sys

from save_data import FrameInfo, TrackInfo


class CollectData:
    def __init__(self, rtsp_url):
        """
        Initialize the Trigger Handler.
=        """

        self.rtsp_url = rtsp_url

        #Load models
        print("Loading models...")
        model_path = 'model/traffic.pt'
        self.plate_model = YOLO(model_path)
        self.ocr_model = PaddleOCR(use_angle_cls=False, use_gpu=True, lang='en', show_log=False)
        print("Models loaded successfully.")

        # Cache to avoid duplicate processing
        self.detected_plates_cache = {}

        # Ensure the directory for saving plates exists
        self.plate_dir = "./plates"
        if not os.path.exists(self.plate_dir):
            os.makedirs(self.plate_dir)

    def process_camera(self):
        """
        Process frames from a specific camera for real-time detection.
        :param cam_index: Index of the camera to process.
        """
        print(f"Starting Trigger for Camera...")
        cap = cv2.VideoCapture(self.rtsp_url)
        track_dict = {}
        
        # Introduce a time-based cache to prevent very recent duplicates
        DUPLICATE_THRESHOLD_SECONDS = 120  # Adjust as needed

        if cap is None or not cap.isOpened():
            print(f"Camera not available!")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame from Camera")
                break
            
            current_time = datetime.datetime.now()
            timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Chỉnh kích thước frame về 1280x720

            # Detect license plates using traffic.pt
            results = self.plate_model.track(source=frame, persist=True, classes=9, conf=0.5, verbose=False, device=0)
            boxes = results[0].boxes

            
            # Get detected IDs and confidence levels
            current_ids = set()
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                bounding_boxes = boxes.xyxy.cpu()
                
                # Clean up expired recent plates
                self.detected_plates_cache = {
                    plate: time for plate, time in self.detected_plates_cache.items() 
                    if (current_time - time).total_seconds() < DUPLICATE_THRESHOLD_SECONDS
                }
                
                # Iterate to get the results
                for box, track_id, conf in zip(bounding_boxes, track_ids, confidences):
                    current_ids.add(track_id)
                    x1, y1, x2, y2 = map(int, box)

                    # Crop the region of interest
                    cropped_img = frame[y1:y2, x1:x2]
                    
                    # Create or update track info with timestamp
                    if track_id not in track_dict:
                        track_dict[track_id] = TrackInfo()
                    
                    # Add new frame to tracking info with timestamp
                    frame_info = FrameInfo(conf, cropped_img, current_time)
                    track_dict[track_id].add_frame(frame_info)
                    track_dict[track_id].active = True

                # Update active status and process inactive tracks
                for track_id in list(track_dict.keys()):
                    if track_id not in current_ids:
                        track_dict[track_id].active = False
                        track_dict[track_id].process_inactive(
                            self.ocr_model, 
                            self.plate_dir, 
                            self.detected_plates_cache
                        )
                        
                        # Enhanced duplicate prevention
                        if track_dict[track_id].text:
                            # Check against permanent cache
                            if (track_dict[track_id].text not in self.detected_plates_cache):
                                
                                # Update caches
                                self.detected_plates_cache[track_dict[track_id].text] = current_time
                                

                # Limit dictionary size
                max_ids = 30
                if len(track_dict) > max_ids:
                    oldest_ids = sorted(track_dict.keys())[:len(track_dict) - max_ids]
                    for old_id in oldest_ids:
                        track_dict.pop(old_id)
            # # Lặp qua các box và vẽ lên frame
            # for box in boxes:
            #     x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ của box
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ hình chữ nhật (green color)

            # frame = cv2.resize(frame, (1280, 720))

            # cv2.imshow("RTSP Camera Stream", frame)

            key = cv2.waitKey(10)
            if key == 27:  # ESC key: quit program
                break
        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()

    def start_processing(self):
        """
        Start a separate thread to process the camera stream.
        """
        self.process_camera()

rtsp_url = 'rtsp://admin1:admin123.@tinhocngoisao.ruijieddns.com:9190/rtsp/streaming?channel=02&subtype=0'

# Create CollectData instance with RTSP URL
collector = CollectData(rtsp_url)
collector.start_processing()
