
import numpy as np
import cv2
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

class Detector:
    def __init__(self, model_path='best.pt'):
        self.use_yolo = False
        self.model = None
        if YOLO_AVAILABLE and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.use_yolo = True
                print("Loaded YOLO model:", model_path)
            except Exception as e:
                print("YOLO load failed:", e)
                self.use_yolo = False
        else:
            print("YOLO not available or model missing - using fallback detector.")
            self.use_yolo = False
            self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)

    def detect(self, frame):
        # frame: BGR image
        if self.use_yolo:
            # ultralytics returns cpu tensors - extract boxes
            results = self.model(frame)[0]
            centroids = []
            for b in results.boxes:
                # b.xyxy gives tensor, convert
                xy = b.xyxy[0].cpu().numpy()
                x1,y1,x2,y2 = xy
                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)
                centroids.append((cx,cy))
            return centroids
        else:
            mask = self.bg.apply(frame)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centroids = []
            for c in cnts:
                if cv2.contourArea(c) < 200: continue
                x,y,w,h = cv2.boundingRect(c)
                centroids.append((int(x+w/2), int(y+h/2)))
            return centroids