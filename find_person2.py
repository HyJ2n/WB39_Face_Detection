import cv2
import face_recognition
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image

def load_vectors(vector_file):
    face_vectors = {}
    with open(vector_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_count = int(parts[0])
            vector = np.array(list(map(float, parts[1:])))
            if frame_count not in face_vectors:
                face_vectors[frame_count] = []
            face_vectors[frame_count].append(vector)
    return face_vectors

def save_vector_to_file(frame_count, vector, vector_file):
    with open(vector_file, 'a') as f:
        vector_str = ','.join(map(str, vector))
        f.write(f"{frame_count},{vector_str}\n")

def compare_and_save_similarities(new_face_vector, saved_face_vectors):
    similarities = {}
    for frame_count, vectors in saved_face_vectors.items():
        for idx, saved_vector in enumerate(vectors):
            distance = np.linalg.norm(new_face_vector - saved_vector)
            similarities[(frame_count, idx)] = distance
    return similarities

def extract_face_vector_and_bbox_from_frame(frame, frame_count, face_folder, vector_file):
    face_locations = face_recognition.face_locations(frame)
    face_vectors = []
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_face_image)
        for face_encoding in face_encodings:
            face_vectors.append(face_encoding)

            # Save face image
            face_filename = os.path.join(face_folder, f"face_{frame_count:04d}_{len(face_vectors):02d}.jpg")
            cv2.imwrite(face_filename, face_image)

            # Save vector to file
            save_vector_to_file(frame_count, face_encoding, vector_file)

    return face_vectors, face_locations

def predict_gender_and_save(model, face_folder, output_folder):
    for face_filename in os.listdir(face_folder):
        face_path = os.path.join(face_folder, face_filename)
        face_image = Image.open(face_path)

        results = model.predict(source=face_image, save=False, conf=0.6)

        if results and len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                label = int(box.cls)
                gender = "Unknown"
                if label == 80:
                    gender = "Male"
                elif label == 81:
                    gender = "Female"

                # Draw bounding box and label on face image
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                label_text = f"{gender}: {confidence:.2f}"
                face_cv2_image = cv2.imread(face_path)
                cv2.rectangle(face_cv2_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(face_cv2_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Save face image with label
                output_face_folder = os.path.join(output_folder, gender)
                if not os.path.exists(output_face_folder):
                    os.makedirs(output_face_folder)
                output_face_filename = os.path.join(output_face_folder, face_filename)
                cv2.imwrite(output_face_filename, face_cv2_image)

def main():
    video_path = r"C:\Users\user\Desktop\사진\KakaoTalk_20240624_153034158.mp4"
    face_folder = r'C:\Users\user\Desktop\wb39_yolov8\gender_model\face_save'
    output_folder = r'C:\Users\user\Desktop\wb39_yolov8\gender_model\face_output'
    vector_file = r'C:\Users\user\Desktop\wb39_yolov8\gender_model\face_vectors.txt'
    
    if not os.path.exists(face_folder):
        os.makedirs(face_folder)
    
    model = YOLO('best.pt')
    
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        face_vectors, face_locations = extract_face_vector_and_bbox_from_frame(frame, frame_count, face_folder, vector_file)
        frame_count += 1
    
    video_capture.release()
    
    predict_gender_and_save(model, face_folder, output_folder)
    print("Processing complete.")

if __name__ == '__main__':
    main()
    