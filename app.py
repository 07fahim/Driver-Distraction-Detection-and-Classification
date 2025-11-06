from flask import Flask, render_template, request, jsonify, send_file
import os
import base64
from werkzeug.utils import secure_filename
import tempfile
import uuid
import cv2
import numpy as np
import onnxruntime as ort

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# =========================
# DEBUG: Check OpenCV FFmpeg support
# =========================
print("OpenCV version:", cv2.__version__)
build_info = cv2.getBuildInformation()
ffmpeg_enabled = "FFMPEG: YES" in build_info
print(f"OpenCV FFmpeg support: {'YES' if ffmpeg_enabled else 'NO'}")

# =========================
# ONNX MODEL SETUP
# =========================
print("Loading ONNX model...")
session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
print("Model loaded successfully!")

# Class names for 12 behaviors
CLASS_NAMES = [
    'c0 - Safe Driving', 'c1 - Texting', 'c2 - Talking on the phone',
    'c3 - Operating the Radio', 'c4 - Drinking', 'c5 - Reaching Behind',
    'c6 - Hair and Makeup', 'c7 - Talking to Passenger',
    'd0 - Eyes Closed', 'd1 - Yawning', 'd2 - Nodding Off', 'd3 - Eyes Open'
]

INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.25

# =========================
# PREPROCESSING & INFERENCE
# =========================
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    img_resized = cv2.resize(img, INPUT_SIZE)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transposed = img_normalized.transpose(2, 0, 1)
    img_batch = np.expand_dims(img_transposed, axis=0)
    return img_batch

def run_inference(image):
    try:
        input_tensor = preprocess_image(image)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        boxes = parse_yolo_output(outputs, image.shape, conf_threshold=CONF_THRESHOLD)
        return boxes
    except Exception as e:
        print(f"Inference error: {e}")
        return []

def parse_yolo_output(outputs, original_shape, conf_threshold=0.25):
    boxes = []
    predictions = outputs[0][0]
    
    orig_h, orig_w = original_shape[:2]
    scale_x = orig_w / INPUT_SIZE[0]
    scale_y = orig_h / INPUT_SIZE[1]
    
    for pred in predictions:
        x1_640, y1_640, x2_640, y2_640, confidence, class_id = pred
        
        if confidence < conf_threshold:
            continue
        
        x1 = int(x1_640 * scale_x)
        y1 = int(y1_640 * scale_y)
        x2 = int(x2_640 * scale_x)
        y2 = int(y2_640 * scale_y)
        
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        class_id_int = int(class_id)
        if class_id_int < 0 or class_id_int >= len(CLASS_NAMES):
            continue
        
        boxes.append({
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'confidence': float(confidence),
            'class_id': class_id_int,
            'class_name': CLASS_NAMES[class_id_int]
        })
    
    return boxes

def draw_boxes_text_overlay(image, boxes, frame_number=None, fps=None):
    annotated = image.copy()
    detection_data = []
    
    if not boxes:
        return annotated, detection_data
    
    boxes_sorted = sorted(boxes, key=lambda b: b['confidence'], reverse=True)
    
    font_scale = 1.0
    font_thickness = 2
    color_bg = (0, 0, 0)
    color_text = (255, 255, 255)
    line_height = 40
    margin = 20
    
    y = annotated.shape[0] - margin
    
    if frame_number is not None and fps is not None:
        fps_text = f"Frame: {frame_number} | FPS: {fps}"
        cv2.putText(annotated, fps_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    for box in boxes_sorted:
        label = f"{box['class_name']}: {box['confidence']:.1%}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        x = margin
        cv2.rectangle(annotated, 
                     (x - 5, y - text_h - 10), 
                     (x + text_w + 5, y + baseline + 5), 
                     color_bg, -1)
        cv2.putText(annotated, label, 
                    (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_text, font_thickness)
        
        y -= line_height
        detection_data.append({
            'class': box['class_name'],
            'confidence': f"{box['confidence']:.1%}",
            'bbox': f"({box['x1']},{box['y1']})-({box['x2']},{box['y2']})"
        })
        
        if y < 0:
            break
    
    return annotated, detection_data

# =========================
# MAX QUALITY H.264 (avc1) + .mp4 WITH ROBUST FALLBACK
# =========================
def create_writer(path, fps, w, h):
    print(f"Creating writer: {path}, fps={fps}, size={w}x{h}")
    
    # Try avc1 (H.264)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    
    if out.isOpened():
        print("Video writer: avc1 (H.264) + .mp4 + QUALITY=100")
        return out

    print("avc1 failed — trying mp4v fallback")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if not out.isOpened():
        raise Exception("VideoWriter failed: both avc1 and mp4v not supported")
    print("Fallback: using mp4v")
    return out

# =========================
# FLASK ROUTES
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image')
def image_page():
    return render_template('image.html')

@app.route('/video')
def video_page():
    return render_template('video.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/api/detect-image', methods=['POST'])
def detect_image():
    try:
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        boxes = run_inference(img)
        annotated, data = draw_boxes_text_overlay(img, boxes)
        _, buf = cv2.imencode('.jpg', annotated)
        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}",
            'detections': data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/detect-video', methods=['POST'])
def detect_video():
    try:
        file = request.files['video']
        upload_id = str(uuid.uuid4())
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{upload_id}_in.mp4")
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{upload_id}.mp4")
        file.save(input_path)

        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened(): raise Exception("Cannot open video")
            fps = max(int(cap.get(cv2.CAP_PROP_FPS)), 1) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            out = create_writer(output_path, fps, w, h)
            cap = cv2.VideoCapture(input_path)
            stats = {}
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret: break
                boxes = run_inference(frame)
                annotated, _ = draw_boxes_text_overlay(frame, boxes, frame_idx, fps)
                out.write(annotated)
                for b in boxes:
                    c = b['class_name']
                    stats.setdefault(c, {'count':0, 'conf':[], 'frames':[]})
                    stats[c]['count'] += 1
                    stats[c]['conf'].append(b['confidence'])
                    stats[c]['frames'].append(frame_idx)
                frame_idx += 1

            cap.release()
            out.release()

            summary = []
            for c, s in stats.items():
                if not s['conf']: continue
                summary.append({
                    'behavior': c,
                    'total_detections': s['count'],
                    'avg_confidence': f"{np.mean(s['conf']):.1%}",
                    'frame_range': f"{min(s['frames'])}-{max(s['frames'])}",
                    'duration': f"{len(set(s['frames']))/fps:.1f}s"
                })
            summary.sort(key=lambda x: x['total_detections'], reverse=True)

            return jsonify({
                'success': True,
                'video_id': upload_id,
                'summary': summary
            })

        finally:
            if os.path.exists(input_path):
                try: os.remove(input_path)
                except: pass

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download-video/<video_id>')
def download_video(video_id):
    try:
        path = os.path.join(app.config['UPLOAD_FOLDER'], f"{secure_filename(video_id)}.mp4")
        if not os.path.exists(path):
            return jsonify({'success': False, 'error': 'Not found'}), 404
        return send_file(path, mimetype='video/mp4', as_attachment=False,
                         download_name='detected_video.mp4')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'codec': 'avc1',
        'container': '.mp4',
        'quality': 'MAX',
        'opencv_ffmpeg': 'YES' if ffmpeg_enabled else 'NO'
    })

if __name__ == '__main__':
    print("SafeDrive AI - MAX QUALITY .mp4 OUTPUT")
    print("Codec: avc1 (H.264) | Container: .mp4 | Quality: 100")
    app.run(debug=True, host='0.0.0.0', port=5000)