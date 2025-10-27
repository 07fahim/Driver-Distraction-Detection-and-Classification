import gradio as gr
import cv2
import numpy as np
import onnxruntime as ort
import tempfile
import pandas as pd

# =========================
# ONNX MODEL SETUP
# =========================
# Load ONNX model with CPU provider for HF Spaces
session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])

# Class names
CLASS_NAMES = [
    'c0 - Safe Driving', 'c1 - Texting', 'c2 - Talking on the phone',
    'c3 - Operating the Radio', 'c4 - Drinking', 'c5 - Reaching Behind',
    'c6 - Hair and Makeup', 'c7 - Talking to Passenger',
    'd0 - Eyes Closed', 'd1 - Yawning', 'd2 - Nodding Off', 'd3 - Eyes Open'
]

# Input size for YOLO model
INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.25

# =========================
# PREPROCESSING
# =========================
def preprocess_image(image):
    """Preprocess image for YOLO ONNX model"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    img_resized = cv2.resize(img, INPUT_SIZE)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transposed = img_normalized.transpose(2, 0, 1)  # HWC to CHW
    img_batch = np.expand_dims(img_transposed, axis=0)  # Add batch dimension
    return img_batch

# =========================
# INFERENCE
# =========================
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
    """
    Parse YOLO ONNX output to bounding boxes
    Output shape: (1, 300, 6) where 6 = [x1, y1, x2, y2, confidence, class_id]
    Coordinates are in pixels relative to 640×640 input image
    """
    boxes = []
    predictions = outputs[0][0]  # Shape: (300, 6)
    
    orig_h, orig_w = original_shape[:2]
    # Scale from 640×640 model input to original image size
    scale_x = orig_w / INPUT_SIZE[0]
    scale_y = orig_h / INPUT_SIZE[1]
    
    for pred in predictions:
        # Format: [x1, y1, x2, y2, confidence, class_id]
        x1_640, y1_640, x2_640, y2_640, confidence, class_id = pred
        
        # Filter by confidence threshold
        if confidence < conf_threshold:
            continue
        
        # Scale coordinates from 640×640 to original image size
        x1 = int(x1_640 * scale_x)
        y1 = int(y1_640 * scale_y)
        x2 = int(x2_640 * scale_x)
        y2 = int(y2_640 * scale_y)
        
        # Ensure valid bounding box within image bounds
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        
        # Skip invalid boxes (zero or negative area)
        if x2 <= x1 or y2 <= y1:
            continue
        
        class_id_int = int(class_id)
        
        # Validate class ID is within range
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

# =========================
# VISUALIZATION
# =========================
def draw_boxes(image, boxes):
    """Draw bounding boxes on image with smart label positioning"""
    annotated = image.copy()
    detection_data = []
    
    # Color map for different classes (consistent colors per class)
    np.random.seed(42)
    colors = [(np.random.randint(50, 255), np.random.randint(50, 255), 
               np.random.randint(50, 255)) for _ in range(len(CLASS_NAMES))]
    
    # Label positioning: top, bottom, middle (cycle through)
    label_positions = ['top', 'bottom', 'middle']
    
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        conf = box['confidence']
        class_id = box['class_id']
        class_name = box['class_name']
        color = colors[class_id]
        
        # Draw rectangle with thicker line
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        # Create label with class name and confidence percentage
        label = f"{class_name} {conf:.1%}"
        
        # Calculate text size for background (larger font)
        font_scale = 0.8
        font_thickness = 2
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Determine label position based on index (cycle through positions)
        position = label_positions[idx % len(label_positions)]
        
        if position == 'top':
            # Position label above the box
            label_y_top = y1 - 10
            label_y_bottom = y1 - label_h - baseline - 10
            
            # If label would go off the top, move it down
            if label_y_bottom < 0:
                label_y_top = y1 + label_h + baseline + 15
                label_y_bottom = y1 + 5
            
            label_x = x1
            text_y = label_y_top - 5
            
        elif position == 'bottom':
            # Position label below the box
            label_y_top = y2 + label_h + baseline + 15
            label_y_bottom = y2 + 5
            
            # If label would go off the bottom, move it up
            if label_y_top > image.shape[0]:
                label_y_top = y2 - 10
                label_y_bottom = y2 - label_h - baseline - 10
            
            label_x = x1
            text_y = label_y_top - 5
            
        else:  # middle
            # Position label in the middle-left of the box
            box_center_y = (y1 + y2) // 2
            label_y_top = box_center_y + (label_h + baseline) // 2
            label_y_bottom = box_center_y - (label_h + baseline) // 2
            
            label_x = x1 + 5
            text_y = label_y_top - 5
        
        # Draw label background rectangle
        cv2.rectangle(annotated, 
                     (label_x, label_y_bottom), 
                     (label_x + label_w + 10, label_y_top), 
                     color, -1)
        
        # Draw label text in white
        cv2.putText(annotated, label, 
                    (label_x + 5, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # Add to DataFrame
        detection_data.append({
            'Class': class_name,
            'Confidence': f"{conf:.1%}",
            'BBox': f"({x1},{y1})-({x2},{y2})"
        })
    
    df = pd.DataFrame(detection_data) if detection_data else pd.DataFrame(
        columns=['Class', 'Confidence', 'BBox'])
    
    return annotated, df

# =========================
# IMAGE/WEBCAM HANDLER
# =========================
def predict_image(image):
    """Process image or webcam capture"""
    if image is None:
        return None, pd.DataFrame(columns=['Class', 'Confidence', 'BBox'])
    
    # Convert RGB to BGR if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    boxes = run_inference(image_bgr)
    result_img, df = draw_boxes(image_bgr, boxes)
    
    # Convert back to RGB for Gradio display
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    return result_img_rgb, df

# =========================
# VIDEO HANDLER
# =========================
def predict_video(video_path):
    """Process video file and return video with detection statistics"""
    if video_path is None:
        return None, pd.DataFrame(columns=['Frame', 'Class', 'Confidence', 'Count'])
    
    try:
        # Create temporary output file
        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, pd.DataFrame(columns=['Frame', 'Class', 'Confidence', 'Count'])
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer with H264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_out.name, fourcc, fps, (w, h))
        
        # Track detections across frames
        all_detections = []
        detection_summary = {}
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            boxes = run_inference(frame)
            annotated, _ = draw_boxes(frame, boxes)
            out.write(annotated)
            
            # Collect detection statistics
            for box in boxes:
                class_name = box['class_name']
                confidence = box['confidence']
                
                # Track for summary
                if class_name not in detection_summary:
                    detection_summary[class_name] = {
                        'count': 0,
                        'avg_confidence': [],
                        'frames': []
                    }
                detection_summary[class_name]['count'] += 1
                detection_summary[class_name]['avg_confidence'].append(confidence)
                detection_summary[class_name]['frames'].append(frame_count)
                
                # Add to detailed list
                all_detections.append({
                    'Frame': frame_count,
                    'Time (s)': f"{frame_count/fps:.2f}",
                    'Class': class_name,
                    'Confidence': f"{confidence:.1%}"
                })
            
            frame_count += 1
            if frame_count % 30 == 0:  # Progress indicator
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        # Create summary DataFrame
        summary_data = []
        for class_name, stats in detection_summary.items():
            avg_conf = np.mean(stats['avg_confidence'])
            frame_range = f"{min(stats['frames'])}-{max(stats['frames'])}"
            summary_data.append({
                'Behavior': class_name,
                'Total Detections': stats['count'],
                'Avg Confidence': f"{avg_conf:.1%}",
                'Frame Range': frame_range,
                'Duration (s)': f"{len(set(stats['frames']))/fps:.1f}"
            })
        
        summary_df = pd.DataFrame(summary_data) if summary_data else pd.DataFrame(
            columns=['Behavior', 'Total Detections', 'Avg Confidence', 'Frame Range', 'Duration (s)'])
        
        # Sort by detection count
        if not summary_df.empty:
            summary_df = summary_df.sort_values('Total Detections', ascending=False)
        
        return temp_out.name, summary_df
    
    except Exception as e:
        print(f"Video processing error: {e}")
        return None, pd.DataFrame(columns=['Behavior', 'Total Detections', 'Avg Confidence', 'Frame Range', 'Duration (s)'])

# =========================
# GRADIO UI
# =========================
with gr.Blocks(theme=gr.themes.Soft(), title="Distracted Driving Detection") as app:
    gr.Markdown(
        """
        # 🚗 Distracted Driving Detection System  
        Detects 12 types of driver distraction behaviors with real-time bounding boxes and confidence scores.
        
        ### Detected Behaviors:
        Safe Driving • Texting • Phone Call • Radio Operation • Drinking • Reaching Behind  
        Hair/Makeup • Talking to Passenger • Eyes Closed • Yawning • Nodding Off • Eyes Open
        """
    )

    with gr.Tab("📷 Image / Webcam"):
        gr.Markdown("### Upload an image or capture from webcam")
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_img = gr.Image(
                    sources=["upload", "webcam"], 
                    type="numpy", 
                    label="Input Image",
                    height=600
                )
                btn_image = gr.Button("🔍 Detect Distractions", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_img = gr.Image(
                    type="numpy", 
                    label="Detection Results",
                    height=600
                )
        
        output_table = gr.DataFrame(
            label="Detection Details", 
            wrap=True,
            row_count=5
        )
        
        btn_image.click(
            fn=predict_image, 
            inputs=input_img, 
            outputs=[output_img, output_table]
        )
        
        # Example images
        gr.Examples(
            examples=[
                ["examples/example1.jpg"],
                ["examples/example2.jpg"],
                ["examples/example3.jpg"],
                ["examples/example4.jpg"],
                ["examples/example5.jpg"],
                ["examples/example6.jpg"],
                ["examples/example7.jpg"],
                ["examples/example8.jpg"],
                ["examples/example9.jpg"],
                ["examples/example10.jpg"],
                ["examples/example11.jpg"],
                ["examples/example12.jpg"],
                
            ],
            inputs=input_img,
            label="📸 Try These Example Images"
        )

    with gr.Tab("🎥 Video Processing"):
        gr.Markdown("### Upload a video file for frame-by-frame analysis")
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_vid = gr.Video(
                    label="Upload Video",
                    height=600
                )
                btn_video = gr.Button("🚗 Process Video", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_vid = gr.Video(
                    label="Processed Output",
                    height=600
                )
        
        # Detection summary table
        output_video_table = gr.DataFrame(
            label="Video Detection Summary",
            wrap=True,
            row_count=10
        )
        
        btn_video.click(
            fn=predict_video, 
            inputs=input_vid, 
            outputs=[output_vid, output_video_table]
        )
        
        # Example videos
        gr.Examples(
            examples=[
                ["examples/video1.mp4"],
                ["examples/video2.mp4"],
            ],
            inputs=input_vid,
            label="🎬 Try These Example Videos"
        )

# =========================
# LAUNCH
# =========================
if __name__ == "__main__":
    # For Hugging Face Spaces deployment
    app.launch(
        share=False,           # No need for share link on HF Spaces
        server_name="0.0.0.0", # Bind to all interfaces for HF Spaces
        server_port=7860       # Default HF Spaces port
    )