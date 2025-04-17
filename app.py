from flask import Flask, request, jsonify
from ultralytics import YOLO
from flask_cors import CORS
import cv2
import numpy as np
import threading
from queue import Queue
import time
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = r"D:\server\best.pt"
MAX_WORKERS = 4  # Adjust based on your GPU memory
BATCH_SIZE = 4   # Optimal batch size for your GPU
INPUT_QUEUE_MAXSIZE = 20  # Prevent memory overload

# Load model with optimized settings
model = YOLO(MODEL_PATH)
model.overrides['batch'] = BATCH_SIZE
model.overrides['verbose'] = False  # Disable verbose output for performance

# Thread-safe queue for request batching
request_queue = Queue(maxsize=INPUT_QUEUE_MAXSIZE)
result_dict = {}
lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def batch_processor():
    """Process requests in batches for optimal GPU utilization"""
    while True:
        batch = []
        batch_ids = []
        
        # Wait for at least one request
        item = request_queue.get()
        batch.append(item['image'])
        batch_ids.append(item['id'])
        
        # Try to gather more requests for the batch (non-blocking)
        while len(batch) < BATCH_SIZE and not request_queue.empty():
            try:
                item = request_queue.get_nowait()
                batch.append(item['image'])
                batch_ids.append(item['id'])
            except:
                break
        
        # Process the batch
        try:
            start_time = time.time()
            batch_results = model(batch)
            
            # Store results
            with lock:
                for img_id, results in zip(batch_ids, batch_results):
                    detections = []
                    for box in results.boxes:
                        detections.append({
                            "class": results.names[int(box.cls)],
                            "confidence": float(box.conf),
                            "bbox": box.xyxy[0].tolist()
                        })
                    result_dict[img_id] = detections
                    
            processing_time = time.time() - start_time
            print(f"✅ Processed batch of {len(batch)} images in {processing_time:.3f}s "
                  f"({len(batch)/processing_time:.1f} FPS)")
                  
        except Exception as e:
            print(f"❌ Batch processing failed: {str(e)}")
            with lock:
                for img_id in batch_ids:
                    result_dict[img_id] = {"error": "Processing failed"}

# Start the batch processing thread
processing_thread = threading.Thread(target=batch_processor, daemon=True)
processing_thread.start()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "🟢 Python inference server is running!",
        "performance": "Optimized for 40+ FPS",
        "config": {
            "max_workers": MAX_WORKERS,
            "batch_size": BATCH_SIZE,
            "queue_size": INPUT_QUEUE_MAXSIZE
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Generate unique ID for this request
        request_id = str(time.time()) + str(threading.get_ident())
        
        # Read and prepare image
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Check if queue is full
        if request_queue.qsize() >= INPUT_QUEUE_MAXSIZE:
            return jsonify({"error": "Server overloaded - try again later"}), 503
            
        # Add to processing queue
        request_queue.put({'id': request_id, 'image': img})
        
        # Wait for results with timeout
        start_time = time.time()
        timeout = 5.0  # seconds
        
        while True:
            with lock:
                if request_id in result_dict:
                    result = result_dict.pop(request_id)
                    break
                
            if time.time() - start_time > timeout:
                with lock:
                    if request_id in result_dict:
                        result = result_dict.pop(request_id)
                    else:
                        return jsonify({"error": "Processing timeout"}), 504
                break
                
            time.sleep(0.005)  # Short sleep to prevent busy waiting
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Disable debug mode for production
    # Consider using a production WSGI server like gunicorn or uWSGI
    app.run(host='192.168.10.117', port=5000, debug=False, threaded=True)