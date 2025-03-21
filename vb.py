from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
import time
from flask_socketio import SocketIO, emit
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


tool = "draw"
selected_tool_index = None
color = (255, 255, 0)
thickness = 4
mask = np.ones((480, 640, 3), dtype="uint8") * 255 
prevx, prevy = 0, 0
var_inits = False 


ml = 150  
max_x, max_y = 300 + ml, 50
tools = np.zeros((max_y, max_x - ml, 3), dtype="uint8")
color_palette = np.zeros((300, 50, 3), dtype="uint8")
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

def init_gui():
    tool_names = ["Line", "Rect", "Draw", "Circle", "Erase", "Color"]
    for idx, name in enumerate(tool_names):
        cv2.putText(tools, name, (10 + idx * 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for i, color in enumerate(colors):
        color_palette[i * 50:(i + 1) * 50, :] = color

init_gui()

def getToolIndex(x):
    if x < 50 + ml:
        return 0  
    elif x < 100 + ml:
        return 1  
    elif x < 150 + ml:
        return 2  
    elif x < 200 + ml:
        return 3  
    elif x < 250 + ml:
        return 4  
    elif x < 300 + ml:
        return 5 
    else:
        return None

def tool_name_by_index(index):
    return ["line", "rectangle", "draw", "circle", "erase", "color_picker"][index]

def index_raised(yi, y9):
    return (y9 - yi) > 40

def generate_frames():
    global tool, color, thickness, prevx, prevy, mask, var_inits, selected_tool_index

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

                if x < max_x and y < max_y and x > ml:
                    index = getToolIndex(x)
                    if index is not None:
                        tool = tool_name_by_index(index)
                        selected_tool_index = index
                        socketio.emit('update_tool', {'tool': tool})

                if tool == "color_picker" and 590 < x < 640 and 50 < y < 350:
                    color = colors[(y - 50) // 50]
                    socketio.emit('update_color', {'color': color})

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

        highlighted_tools = tools.copy()
        if selected_tool_index is not None:
            cv2.rectangle(
                highlighted_tools,
                (selected_tool_index * 50, 0),
                (selected_tool_index * 50 + 50, max_y),
                (0, 255, 0),
                2
            )

        blended = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)
        blended[:max_y, ml:max_x] = cv2.addWeighted(highlighted_tools, 0.9, blended[:max_y, ml:max_x], 0.1, 0)
        blended[50:350, 590:640] = color_palette

        _, buffer = cv2.imencode('.jpg', blended)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_image_png', methods=['POST'])
def save_image_png():
    global mask
    os.makedirs("static", exist_ok=True)
    timestamp = int(time.time())
    save_path = f"static/drawing_{timestamp}.png"
    cv2.imwrite(save_path, mask)
    return jsonify({"image": f"/{save_path}"})

@app.route('/save_image_jpg', methods=['POST'])
def save_image_jpg():
    global mask
    os.makedirs("static", exist_ok=True)
    timestamp = int(time.time())
    save_path = f"static/drawing_{timestamp}.jpg"
    cv2.imwrite(save_path, mask)
    return jsonify({"image": f"/{save_path}"})

@app.route('/save_image_pdf', methods=['POST'])
def save_image_pdf():
    global mask
    os.makedirs("static", exist_ok=True)
    timestamp = int(time.time())
    save_path_img = f"static/drawing_{timestamp}.png"
    save_path_pdf = f"static/drawing_{timestamp}.pdf"
    cv2.imwrite(save_path_img, mask)
    image = Image.open(save_path_img).convert('RGB')
    image.save(save_path_pdf)
    return jsonify({"pdf": f"/{save_path_pdf}"})

@app.route('/clear_canvas', methods=['POST'])
def clear_canvas():
    global mask
    mask = np.ones((480, 640, 3), dtype="uint8") * 255
    return jsonify({"status": "canvas cleared"})

if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0')
