# app.py
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import threading
from queue import Queue
import base64
import json

app = Flask(__name__)

# Global variables
camera = None
output_frame = None
lock = threading.Lock()
model = None
detection_queue = Queue(maxsize=1)

def initialize_model():
    try:
        model = YOLO("C:/Users/bhavi/Downloads/drone_detection_web/yolov8n_trained.pt")  # Update path as needed
        print("Model loaded successfully")
        print(f"Model classes: {model.names}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def draw_multiple_bounding_boxes(frame, x1, y1, x2, y2, score, color_main=(0, 255, 0)):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_main, 2)
    
    # Inner dashed box
    dash_length = 10
    gap_length = 5
    x, y = x1 + 5, y1 + 5
    w, h = x2 - x1 - 10, y2 - y1 - 10
    
    pts = []
    for i in range(0, w, dash_length + gap_length):
        pts.extend([(x + i, y), (x + min(i + dash_length, w), y)])
        pts.extend([(x + i, y + h), (x + min(i + dash_length, w), y + h)])
    for i in range(0, h, dash_length + gap_length):
        pts.extend([(x, y + i), (x, y + min(i + dash_length, h))])
        pts.extend([(x + w, y + i), (x + w, y + min(i + dash_length, h))])
    
    for i in range(0, len(pts), 2):
        if i + 1 < len(pts):
            cv2.line(frame, pts[i], pts[i+1], (255, 255, 255), 1)

def draw_confidence_bars(frame, x1, y1, x2, y2, score):
    bar_width = 100
    bar_height = 10
    bar_x = x1
    bar_y = y1 - 30
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), -1)
    
    conf_width = int(bar_width * score)
    conf_color = (0, int(255 * score), int(255 * (1 - score)))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), conf_color, -1)
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)

def process_frame(frame):
    global model
    
    if model is None:
        model = initialize_model()
        if model is None:
            return frame
    
    threshold = 0.5
    
    try:
        results = model(frame, verbose=False)[0]
        detections = []
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            
            if score > threshold:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                draw_multiple_bounding_boxes(frame, x1, y1, x2, y2, score)
                draw_confidence_bars(frame, x1, y1, x2, y2, score)
                
                class_name = model.names[int(class_id)]
                label = f"{class_name}: {score*100:.2f}%"
                
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_height - 45),
                    (x1 + label_width, y1 - 35),
                    (0, 0, 0),
                    -1,
                )
                
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                
                detections.append({
                    'class': class_name,
                    'confidence': float(score),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        # Update detection queue
        try:
            detection_queue.get_nowait()  # Remove old detection if exists
        except:
            pass
        detection_queue.put(detections)
        
    except Exception as e:
        print(f"Error during inference: {e}")
    
    return frame

def generate_frames():
    global output_frame, lock, camera
    
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame = process_frame(frame)
        
        with lock:
            output_frame = frame.copy()
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detections')
def get_detections():
    try:
        detections = detection_queue.get_nowait()
        return jsonify(detections)
    except:
        return jsonify([])

if __name__ == '__main__':
    model = initialize_model()
    app.run(debug=True)