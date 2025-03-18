from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import threading
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# Video capture
cap = cv2.VideoCapture(0)

# Set camera resolution to 640x480 (matching older code)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Drawing variables
tool = "draw"
color = (255, 255, 0)
thickness = 4
mask = np.ones((480, 640, 3), dtype="uint8") * 255  # White background
prevx, prevy = 0, 0
var_inits = False  # For shape initialization

# Toolbar and color palette
ml = 150  # Margin left for toolbar
max_x, max_y = 300 + ml, 50
tools = np.zeros((max_y, max_x - ml, 3), dtype="uint8")
color_palette = np.zeros((300, 50, 3), dtype="uint8")
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

# Initialize toolbar and color palette
def init_gui():
    cv2.putText(tools, "Line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(tools, "Rect", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(tools, "Draw", (110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(tools, "Circle", (160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(tools, "Erase", (210, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(tools, "Color", (260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for i, color in enumerate(colors):
        color_palette[i * 50:(i + 1) * 50, :] = color

init_gui()

# Tool selection logic
def getTool(x):
    if x < 50 + ml:
        return "line"
    elif x < 100 + ml:
        return "rectangle"
    elif x < 150 + ml:
        return "draw"
    elif x < 200 + ml:
        return "circle"
    elif x < 250 + ml:
        return "erase"
    elif x < 300 + ml:
        return "color_picker"
    else:
        return "none"

# Index finger raised check
def index_raised(yi, y9):
    return (y9 - yi) > 40

# Generate video frames
def generate_frames():
    global tool, color, thickness, prevx, prevy, mask, var_inits

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for landmarks in result.multi_hand_landmarks:
                draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                x, y = int(landmarks.landmark[8].x * 640), int(landmarks.landmark[8].y * 480)
                xi, yi = int(landmarks.landmark[12].x * 640), int(landmarks.landmark[12].y * 480)
                y9 = int(landmarks.landmark[9].y * 480)

                # Tool selection
                if x < max_x and y < max_y and x > ml:
                    tool = getTool(x)
                    socketio.emit('update_tool', {'tool': tool})

                # Color selection
                if tool == "color_picker" and 590 < x < 640 and 50 < y < 350:
                    color = colors[(y - 50) // 50]
                    socketio.emit('update_color', {'color': color})

                # Drawing logic
                if index_raised(yi, y9):
                    if tool == "draw":
                        cv2.line(mask, (prevx, prevy), (x, y), color, thickness)
                        prevx, prevy = x, y
                    elif tool == "line":
                        if not var_inits:
                            xii, yii = x, y
                            var_inits = True
                        cv2.line(frame, (xii, yii), (x, y), color, thickness)
                    elif tool == "rectangle":
                        if not var_inits:
                            xii, yii = x, y
                            var_inits = True
                        cv2.rectangle(frame, (xii, yii), (x, y), color, thickness)
                    elif tool == "circle":
                        if not var_inits:
                            xii, yii = x, y
                            var_inits = True
                        radius = int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5)
                        cv2.circle(frame, (xii, yii), radius, color, thickness)
                    elif tool == "erase":
                        cv2.circle(mask, (x, y), 30, (255, 255, 255), -1)
                else:
                    if var_inits:
                        if tool == "line":
                            cv2.line(mask, (xii, yii), (x, y), color, thickness)
                        elif tool == "rectangle":
                            cv2.rectangle(mask, (xii, yii), (x, y), color, thickness)
                        elif tool == "circle":
                            radius = int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5)
                            cv2.circle(mask, (xii, yii), radius, color, thickness)
                        var_inits = False
                    prevx, prevy = x, y

        # Compose final frame
        blended = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)
        blended[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, blended[:max_y, ml:max_x], 0.3, 0)
        blended[50:350, 590:640] = color_palette

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', blended)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# SocketIO events
@socketio.on('update_tool')
def handle_update_tool(data):
    global tool
    tool = data['tool']

@socketio.on('update_color')
def handle_update_color(data):
    global color
    color = tuple(data['color'])

@socketio.on('update_thickness')
def handle_update_thickness(data):
    global thickness
    thickness = data['thickness']

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')