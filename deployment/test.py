import cv2
import numpy as np
import onnxruntime as ort

# Load model
session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])

# Load test image
image = cv2.imread("2.jpg")  # Replace with your test image
original_shape = image.shape

# Preprocess
img_resized = cv2.resize(image, (640, 640))
img_normalized = img_resized.astype(np.float32) / 255.0
img_transposed = img_normalized.transpose(2, 0, 1)
img_batch = np.expand_dims(img_transposed, axis=0)

# Run inference
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: img_batch})

print("Output shape:", outputs[0].shape)
print("\nFirst 5 detections:")
print(outputs[0][0][:5])

# Check coordinate ranges
detections = outputs[0][0]
valid_detections = detections[detections[:, 4] > 0.25]  # Filter by confidence

print(f"\nValid detections (conf > 0.25): {len(valid_detections)}")

if len(valid_detections) > 0:
    print("\nCoordinate ranges:")
    print(f"x1 range: [{valid_detections[:, 0].min():.3f}, {valid_detections[:, 0].max():.3f}]")
    print(f"y1 range: [{valid_detections[:, 1].min():.3f}, {valid_detections[:, 1].max():.3f}]")
    print(f"x2 range: [{valid_detections[:, 2].min():.3f}, {valid_detections[:, 2].max():.3f}]")
    print(f"y2 range: [{valid_detections[:, 3].min():.3f}, {valid_detections[:, 3].max():.3f}]")
    print(f"Confidence range: [{valid_detections[:, 4].min():.3f}, {valid_detections[:, 4].max():.3f}]")
    print(f"Class IDs: {np.unique(valid_detections[:, 5].astype(int))}")

# Visualize one detection
if len(valid_detections) > 0:
    det = valid_detections[0]
    x1, y1, x2, y2, conf, cls = det
    
    print(f"\nFirst detection:")
    print(f"  Raw coords: x1={x1:.3f}, y1={y1:.3f}, x2={x2:.3f}, y2={y2:.3f}")
    
    # Check if coordinates are normalized
    if x2 <= 1.0 and y2 <= 1.0:
        print("  ✓ Coordinates appear to be NORMALIZED (0-1)")
        # Scale to original image size
        orig_h, orig_w = original_shape[:2]
        x1_px = int(x1 * orig_w)
        y1_px = int(y1 * orig_h)
        x2_px = int(x2 * orig_w)
        y2_px = int(y2 * orig_h)
    else:
        print("  ✓ Coordinates appear to be in PIXELS")
        # Already in pixels, but may need scaling from 640x640 to original
        scale_x = original_shape[1] / 640
        scale_y = original_shape[0] / 640
        x1_px = int(x1 * scale_x)
        y1_px = int(y1 * scale_y)
        x2_px = int(x2 * scale_x)
        y2_px = int(y2 * scale_y)
    
    print(f"  Pixel coords: ({x1_px}, {y1_px}) - ({x2_px}, {y2_px})")
    print(f"  Confidence: {conf:.3f}")
    print(f"  Class ID: {int(cls)}")
    
    # Draw and save
    result = image.copy()
    cv2.rectangle(result, (x1_px, y1_px), (x2_px, y2_px), (0, 255, 0), 2)
    cv2.putText(result, f"Class {int(cls)}: {conf:.2f}", 
                (x1_px, y1_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite("test_output.jpg", result)
    print("\n✓ Saved visualization to test_output.jpg")
    
    
    

                