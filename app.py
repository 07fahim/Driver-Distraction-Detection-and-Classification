from flask import Flask, render_template, request, jsonify, send_file
import os
import base64
from werkzeug.utils import secure_filename
import tempfile
import uuid
import shutil
import cv2
import numpy as np
import onnxruntime as ort

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# =========================
# ONNX MODEL SETUP
# =========================
print("Loading ONNX model...")
print("💻 Using CPU for inference (deployment-optimized)")
session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
print("✅ Model loaded successfully!")

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
    """Preprocess image for YOLO ONNX model"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    img_resized = cv2.resize(img, INPUT_SIZE)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transposed = img_normalized.transpose(2, 0, 1)  # HWC to CHW
    img_batch = np.expand_dims(img_transposed, axis=0)  # Add batch dimension
    return img_batch

def run_inference(image):
    """Run ONNX inference on image"""
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
    """Parse YOLO ONNX output to bounding boxes"""
    boxes = []
    predictions = outputs[0][0]  # Shape: (300, 6)
    
    orig_h, orig_w = original_shape[:2]
    scale_x = orig_w / INPUT_SIZE[0]
    scale_y = orig_h / INPUT_SIZE[1]
    
    for pred in predictions:
        x1_640, y1_640, x2_640, y2_640, confidence, class_id = pred
        
        if confidence < conf_threshold:
            continue
        
        # Scale coordinates
        x1 = int(x1_640 * scale_x)
        y1 = int(y1_640 * scale_y)
        x2 = int(x2_640 * scale_x)
        y2 = int(y2_640 * scale_y)
        
        # Ensure valid bounding box
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
    """
    Draw TEXT OVERLAYS at bottom of image (like Gradio code)
    NO bounding boxes, just text labels with confidence
    """
    annotated = image.copy()
    detection_data = []
    
    if not boxes:
        return annotated, detection_data
    
    # Sort boxes by confidence (highest first)
    boxes_sorted = sorted(boxes, key=lambda b: b['confidence'], reverse=True)
    
    # Text styling
    font_scale = 1.0
    font_thickness = 2
    color_bg = (0, 0, 0)  # Black background
    color_text = (255, 255, 255)  # White text
    line_height = 40
    margin = 20
    
    # Start from bottom of image
    y = annotated.shape[0] - margin
    
    # Add FPS counter in top-left (for videos)
    if frame_number is not None and fps is not None:
        fps_text = f"Frame: {frame_number} | FPS: {fps}"
        cv2.putText(annotated, fps_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw each detection as text overlay at bottom
    for box in boxes_sorted:
        label = f"{box['class_name']}: {box['confidence']:.1%}"
        
        # Calculate text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        x = margin
        
        # Draw black background rectangle
        cv2.rectangle(annotated, 
                     (x - 5, y - text_h - 10), 
                     (x + text_w + 5, y + baseline + 5), 
                     color_bg, -1)
        
        # Draw white text
        cv2.putText(annotated, label, 
                    (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_text, font_thickness)
        
        # Move up for next line
        y -= line_height
        
        # Collect detection data
        detection_data.append({
            'class': box['class_name'],
            'confidence': f"{box['confidence']:.1%}",
            'bbox': f"({box['x1']},{box['y1']})-({box['x2']},{box['y2']})"
        })
        
        # Stop if we run out of space
        if y < 0:
            break
    
    return annotated, detection_data

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
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'}), 400
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image'}), 400
        
        # Run detection
        boxes = run_inference(image)
        
        # Draw TEXT OVERLAY (no bounding boxes)
        annotated_image, detection_data = draw_boxes_text_overlay(image, boxes)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{img_base64}",
            'detections': detection_data
        })
    
    except Exception as e:
        print(f"Error in detect_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/detect-video', methods=['POST'])
def detect_video():
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No video selected'}), 400
        
        # Save uploaded video
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process video
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_output.mp4")
            summary_data = process_video(filepath, output_path)
            
            # Generate video ID
            video_id = str(uuid.uuid4())
            final_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}.mp4")
            shutil.move(output_path, final_path)
            
            return jsonify({
                'success': True,
                'video_id': video_id,
                'summary': summary_data
            })
        
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        print(f"Error in detect_video: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

def process_video(input_path, output_path):
    """Process video with TEXT OVERLAY style (like Gradio)"""
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise Exception("Cannot open video file")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer with codec fallback
    fourcc = None
    out = None
    codecs_to_try = [
        ('avc1', 'H.264'),      # Preferred
        ('mp4v', 'MPEG-4'),     # Fallback 1
        ('XVID', 'Xvid'),       # Fallback 2
        ('MJPG', 'Motion JPEG') # Fallback 3
    ]
    
    for codec_name, codec_desc in codecs_to_try:
        try:
            print(f"Trying codec: {codec_desc} ({codec_name})")
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            if out.isOpened():
                print(f"✅ Successfully initialized with {codec_desc}")
                break
            else:
                print(f"❌ {codec_desc} failed to initialize")
                out.release()
                out = None
        except Exception as e:
            print(f"❌ {codec_desc} error: {e}")
            if out:
                out.release()
            out = None
    
    if out is None or not out.isOpened():
        cap.release()
        raise Exception("Could not initialize video writer with any codec")
    
    # Track detections
    detection_summary = {}
    frame_count = 0
    
    print(f"Processing video: {total_frames} frames at {fps} FPS")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            boxes = run_inference(frame)
            
            # Draw TEXT OVERLAY (with frame counter)
            annotated, _ = draw_boxes_text_overlay(frame, boxes, frame_count, fps)
            
            # Write frame
            success = out.write(annotated)
            if not success:
                print(f"Warning: Failed to write frame {frame_count}")
            
            # Collect statistics
            for box in boxes:
                class_name = box['class_name']
                confidence = box['confidence']
                
                if class_name not in detection_summary:
                    detection_summary[class_name] = {
                        'count': 0,
                        'confidences': [],
                        'frames': []
                    }
                
                detection_summary[class_name]['count'] += 1
                detection_summary[class_name]['confidences'].append(confidence)
                detection_summary[class_name]['frames'].append(frame_count)
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
    
    finally:
        cap.release()
        out.release()
    
    # Verify output file was created
    if not os.path.exists(output_path):
        raise Exception(f"Output video file was not created: {output_path}")
    
    if os.path.getsize(output_path) == 0:
        raise Exception(f"Output video file is empty: {output_path}")
    
    # Create summary
    summary_data = []
    for class_name, stats in detection_summary.items():
        avg_conf = np.mean(stats['confidences'])
        frame_range = f"{min(stats['frames'])}-{max(stats['frames'])}"
        duration = len(set(stats['frames'])) / fps
        
        summary_data.append({
            'behavior': class_name,
            'total_detections': stats['count'],
            'avg_confidence': f"{avg_conf:.1%}",
            'frame_range': frame_range,
            'duration': f"{duration:.1f}s"
        })
    
    # Sort by detection count
    summary_data.sort(key=lambda x: x['total_detections'], reverse=True)
    
    print(f"Video processing complete: {len(summary_data)} behaviors detected")
    print(f"Output file size: {os.path.getsize(output_path)} bytes")
    
    return summary_data

@app.route('/api/download-video/<video_id>')
def download_video(video_id):
    try:
        video_id = secure_filename(video_id)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}.mp4")
        
        if not os.path.exists(video_path):
            return jsonify({'success': False, 'error': 'Video not found'}), 404
        
        return send_file(
            video_path, 
            mimetype='video/mp4', 
            as_attachment=False,
            download_name='detected_video.mp4'
        )
    
    except Exception as e:
        print(f"Error in download_video: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'Local ONNX',
        'style': 'Text Overlay (Gradio Style)',
        'model_loaded': session is not None
    })

if __name__ == '__main__':
    print(f"Model: best.onnx")
    print(f"Classes: {len(CLASS_NAMES)}")
    app.run(debug=True, host='0.0.0.0', port=5000)