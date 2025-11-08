from flask import Flask, render_template, request, jsonify, send_file
from gradio_client import Client, handle_file
from pathlib import Path
import base64
import shutil
from werkzeug.utils import secure_filename
import uuid
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['RESULTS_FOLDER'] = '/tmp/results'

upload_dir = Path(app.config['UPLOAD_FOLDER'])
result_dir = Path(app.config['RESULTS_FOLDER'])

upload_dir.mkdir(parents=True, exist_ok=True)
result_dir.mkdir(parents=True, exist_ok=True)

# Initialize Gradio Client
HF_SPACE = "yeager07/distracted-driving-detection"
client = None

def get_client():
    """Lazy load client to avoid startup issues"""
    global client
    if client is None:
        try:
            client = Client(HF_SPACE)
        except Exception as e:
            print(f"Client init warning: {e}")
    return client

# =========================
# ROUTES
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

# =========================
# IMAGE DETECTION API
# =========================
@app.route('/api/detect-image', methods=['POST'])
def detect_image():
    temp_path = None
    output_image_path = None
    
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400
        
        # Save uploaded file to /tmp/uploads
        filename = secure_filename(file.filename)
        temp_path = upload_dir / f"{uuid.uuid4()}_{filename}"
        file.save(str(temp_path))
        
        try:
            # Get or initialize client
            api_client = get_client()
            if api_client is None:
                return jsonify({'success': False, 'error': 'API service unavailable'}), 503
            
            # Call Hugging Face API
            result = api_client.predict(
                image=handle_file(str(temp_path)),
                api_name="/predict_image"
            )
            
            # Parse result
            output_image_path = Path(result[0])
            detection_table = result[1]
            
            # Read output image as base64
            with open(output_image_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Parse detections
            detections = []
            if detection_table and isinstance(detection_table, dict) and 'data' in detection_table:
                for row in detection_table['data']:
                    if len(row) >= 2:
                        try:
                            confidence_str = row[1].rstrip('%').strip()
                            confidence_val = float(confidence_str)
                            detections.append({
                                'class': row[0],
                                'confidence': f"{confidence_val:.1f}%"
                            })
                        except (ValueError, AttributeError):
                            detections.append({
                                'class': row[0],
                                'confidence': 'N/A'
                            })
            
            return jsonify({
                'success': True,
                'image': f"data:image/jpeg;base64,{img_data}",
                'detections': detections
            })
        
        finally:
            # Cleanup
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            if output_image_path and output_image_path.exists():
                output_image_path.unlink(missing_ok=True)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# =========================
# VIDEO DETECTION API
# =========================
@app.route('/api/detect-video', methods=['POST'])
def detect_video():
    temp_path = None
    
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video uploaded'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400
        
        # Save video
        filename = secure_filename(file.filename)
        video_id = str(uuid.uuid4())
        temp_path = upload_dir / f"{video_id}_{filename}"
        file.save(str(temp_path))
        
        try:
            # Get or initialize client
            api_client = get_client()
            if api_client is None:
                return jsonify({'success': False, 'error': 'API service unavailable'}), 503
            
            # Call HF API
            result = api_client.predict(
                video_path={"video": handle_file(str(temp_path))},
                api_name="/predict_video"
            )
            
            output_video = result[0]
            summary_table = result[1]
            
            # Extract video path
            if isinstance(output_video, dict) and 'video' in output_video:
                output_video_path = Path(output_video['video'])
            else:
                output_video_path = Path(output_video)
            
            # Save result
            result_path = result_dir / f"{video_id}_result.mp4"
            shutil.copy(str(output_video_path), str(result_path))
            
            # Parse summary
            summary = []
            if summary_table and isinstance(summary_table, dict) and 'data' in summary_table:
                for row in summary_table['data']:
                    if len(row) >= 5:
                        summary.append({
                            'behavior': row[0],
                            'total_detections': row[1],
                            'avg_confidence': row[2],
                            'frame_range': row[3],
                            'duration': row[4]
                        })
            
            return jsonify({
                'success': True,
                'video_id': video_id,
                'summary': summary,
                'browser_playback': 'available'
            })
        
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# =========================
# VIDEO DOWNLOAD API
# =========================
@app.route('/api/download-video/<video_id>')
def download_video(video_id):
    try:
        result_path = result_dir / f"{video_id}_result.mp4"
        
        if not result_path.exists():
            return jsonify({'error': 'Video not found'}), 404
        
        return send_file(str(result_path), mimetype='video/mp4', as_attachment=False)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =========================
# CLEANUP TASK
# =========================
def cleanup_old_files():
    """Remove files older than 1 hour"""
    import time
    current_time = time.time()
    
    for folder in [upload_dir, result_dir]:
        if not folder.exists():
            continue
        for filepath in folder.iterdir():
            if filepath.is_file():
                file_age = current_time - filepath.stat().st_mtime
                if file_age > 3600:
                    try:
                        filepath.unlink()
                    except:
                        pass

# =========================
# ERROR HANDLERS
# =========================
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 100MB)'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    cleanup_old_files()
    app.run(debug=True, host='0.0.0.0', port=5000)