import cv2
from ultralytics import YOLO
from collections import defaultdict
import time
import numpy as np
import matplotlib.pyplot as plt
import yt_dlp 


VIDEO_SOURCES = ['https://www.youtube.com/live/y-Os52eW2rg?si=ra6Jy4zIJ_FsxCR7',
    'https://www.youtube.com/live/qMYlpMsWsBE?si=5TFijJwkh4qQVebm',
    
]


# VIDEO_SOURCES = ['traffic2.mp4']


# VIDEO_SOURCES = [0]

MODEL_PATH = 'yolov8n.pt'      # Standard YOLOv8 model

# MODEL_PATH = 'yolov8n.onnx'
#  MODEL_PATH = 'yolov8n.engine'



#
# print("Initializing model for export...")
# model_for_export = YOLO('yolov8n.pt') # Always export from the base .pt model
# print("Exporting model to ONNX format... This may take a moment.")
# model_for_export.export(format='onnx') # Creates 'yolov8n.onnx'
# print("Exporting model to TensorRT format... This will take several minutes.")
# model_for_export.export(format='engine') # Creates 'yolov8n.engine'
# print("Model export complete.")



final_video_source = None
for source in VIDEO_SOURCES:
    if isinstance(source, str) and ("youtube.com" in source or "youtu.be" in source):
        print(f"Attempting YouTube URL: {source}")
        try:
            ydl_opts = {'format': 'best', 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(source, download=False)
                direct_url = info_dict.get('url', None)
                if direct_url:
                    print(f"Success! Found direct stream URL.")
                    final_video_source = direct_url
                    break 
        except Exception as e:
            
            print(f"  -> Note: Stream is unavailable. Trying next source.")
            continue
    else:
       
        print(f"Attempting local source: {source}")
        final_video_source = source
        break

if not final_video_source:
    print("FATAL ERROR: Could not open any of the provided video sources. Please check the URLs or file paths.")
    exit()



print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

class_names = model.names


TARGET_CLASSES = ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle']


validated_classes = []
for cls in TARGET_CLASSES:
    if cls in class_names.values():
        validated_classes.append(cls)
    else:
        print(f"Warning: Class '{cls}' is not in this model's vocabulary and will be ignored.")
TARGET_CLASS_IDS = [k for k, v in class_names.items() if v in validated_classes]
print(f"Tracking the following valid classes: {validated_classes}")



print(f"Opening final video source...")
cap = cv2.VideoCapture(final_video_source)
if not cap.isOpened():
    print(f"Error: Could not open final video source.")
    exit()


track_history = defaultdict(lambda: [])

crossed_ids_up = set()
crossed_ids_down = set()

class_counts_up = defaultdict(int)
class_counts_down = defaultdict(int)

plot_data = defaultdict(list)
frame_count = 0

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30 

line_y = int(video_height * 0.5)
counting_line = [(0, line_y), (video_width, line_y)]


start_time = time.time()

print("Processing video... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Stream ended or failed to read frame.")
        break

    frame_count += 1

   
    results = model.track(frame, persist=True, verbose=False, classes=TARGET_CLASS_IDS)

 
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        annotated_frame = results[0].plot()

     
        cv2.line(annotated_frame, counting_line[0], counting_line[1], (0, 255, 0), 3)

      
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x, y, w, h = box
            center_x, center_y = int(x), int(y)
            class_name = class_names[class_id]
            
            track = track_history[track_id]
            track.append((center_x, center_y))

        
            if len(track) > 1:
                prev_y = track[-2][1]
                curr_y = track[-1][1]

             
                if prev_y < line_y and curr_y >= line_y and track_id not in crossed_ids_down:
                    crossed_ids_down.add(track_id)
                    class_counts_down[class_name] += 1
                
           
                elif prev_y > line_y and curr_y <= line_y and track_id not in crossed_ids_up:
                    crossed_ids_up.add(track_id)
                    class_counts_up[class_name] += 1
    else:
       
        annotated_frame = frame

    
    info_box_height = 25 + (len(validated_classes) * 20) * 2 + 20
    cv2.rectangle(annotated_frame, (0, 0), (250, info_box_height), (0,0,0), -1)

    y_offset = 20
    cv2.putText(annotated_frame, "Moving Down", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_offset += 20
    for name in validated_classes:
        count = class_counts_down[name]
        cv2.putText(annotated_frame, f"- {name.capitalize()}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        y_offset += 20

    y_offset += 10
    cv2.putText(annotated_frame, "Moving Up", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    y_offset += 20
    for name in validated_classes:
        count = class_counts_up[name]
        cv2.putText(annotated_frame, f"- {name.capitalize()}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        y_offset += 20


    if frame_count % int(fps) == 0:
        current_time_sec = frame_count / fps
        total_down = sum(class_counts_down.values())
        total_up = sum(class_counts_up.values())
        plot_data['time'].append(current_time_sec)
        plot_data['down'].append(total_down)
        plot_data['up'].append(total_up)

    cv2.imshow("Advanced Traffic Analyzer", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

end_time = time.time()
cap.release()
cv2.destroyAllWindows()

if frame_count > 0:
    processing_time = end_time - start_time
    avg_fps = frame_count / processing_time
    print("\n--- BENCHMARKING RESULTS ---")
    print(f"Model Used: {MODEL_PATH}")
    print(f"Total Frames Processed: {frame_count}")
    print(f"Total Processing Time: {processing_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print("--------------------------\n")
else:
    print("No frames were processed.")


if plot_data['time']:
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(plot_data['time'], plot_data['down'], label='Total Objects Moving Down', color='cyan', marker='o', linestyle='-')
    ax.plot(plot_data['time'], plot_data['up'], label='Total Objects Moving Up', color='magenta', marker='x', linestyle='--')
    
    ax.set_title('Object Count Over Time', fontsize=16)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Cumulative Object Count', fontsize=12)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    print("Displaying plot. Close the plot window to exit the script.")
    plt.show()
else:
    print("No data collected for plotting (video might be too short or no objects crossed the line).")

