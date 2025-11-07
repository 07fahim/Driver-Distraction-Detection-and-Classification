from flask import Flask, render_template, request, jsonify, send_file
from gradio_client import Client, handle_file
import os
import base64
import shutil
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['RESULTS_FOLDER'] = 'temp_results'


# Initialize Gradio Client
HF_SPACE = "yeager07/distracted-driving-detection"
client = Client(HF_SPACE)

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
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
        file.save(temp_path)
        
        try:
            # Call Hugging Face API
            result = client.predict(
                image=handle_file(temp_path),
                api_name="/predict_image"
            )
            
            # Parse result - result is a tuple: (image_path, dataframe_dict)
            output_image_path = result[0]  # This is a string path
            detection_table = result[1]  # This is a dict with 'headers' and 'data'
            
            # Convert output image to base64
            with open(output_image_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Parse detections from dataframe — NO BBOX, FIX FLOAT ERROR
            detections = []
            if detection_table and isinstance(detection_table, dict) and 'data' in detection_table:
                for row in detection_table['data']:
                    if len(row) >= 2:
                        try:
                            # Remove % and convert to float
                            confidence_str = row[1].rstrip('%').strip()
                            confidence_val = float(confidence_str)
                            detections.append({
                                'class': row[0],
                                'confidence': f"{confidence_val:.1f}%"
                            })
                        except (ValueError, AttributeError):
                            # Fallback if confidence is not a string/number
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
            # Cleanup temp files
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if isinstance(result[0], str) and os.path.exists(result[0]):
                try:
                    os.remove(result[0])
                except:
                    pass
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# =========================
# VIDEO DETECTION API
# =========================
@app.route('/api/detect-video', methods=['POST'])
def detect_video():
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video uploaded'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        video_id = str(uuid.uuid4())
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}_{filename}")
        file.save(temp_path)
        
        try:
            # Call Hugging Face API
            result = client.predict(
                video_path={"video": handle_file(temp_path)},
                api_name="/predict_video"
            )
            
            # Parse result - result is a tuple: (video_dict, dataframe_dict)
            output_video = result[0]  # This is a dict with 'video' key
            summary_table = result[1]  # DataFrame dict with 'headers' and 'data'
            
            # Extract video path from dict
            if isinstance(output_video, dict) and 'video' in output_video:
                output_video_path = output_video['video']
            else:
                output_video_path = output_video  # Fallback if it's a string
            
            # Save output video to results folder
            result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{video_id}_result.mp4")
            shutil.copy(output_video_path, result_path)
            
            # Parse summary from dataframe
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
            # Cleanup upload file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# =========================
# VIDEO DOWNLOAD API
# =========================
@app.route('/api/download-video/<video_id>')
def download_video(video_id):
    try:
        # Find the result file
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{video_id}_result.mp4")
        
        if not os.path.exists(result_path):
            return jsonify({'error': 'Video not found'}), 404
        
        return send_file(result_path, mimetype='video/mp4', as_attachment=False)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =========================
# CLEANUP TASK (Optional)
# =========================
def cleanup_old_files():
    """Remove files older than 1 hour"""
    import time
    current_time = time.time()
    
    for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > 3600:  # 1 hour
                    try:
                        os.remove(filepath)
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
    # Periodic cleanup (run in production with scheduler like APScheduler)
    cleanup_old_files()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)