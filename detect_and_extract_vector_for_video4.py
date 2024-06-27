import cv2
import face_recognition
import os
from multiprocessing import Pool, cpu_count, freeze_support

def process_frame(frame_path):
    frame_count, frame = frame_path

    # Convert frame to RGB (required by face_recognition library)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)

    # Encode faces to get face encodings
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    return frame_count, face_locations, face_encodings

def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_dir, exist_ok=True)

    frames_to_process = [(i, cap.read()[1]) for i in range(total_frames)]

    with Pool(cpu_count()) as pool:
        results = pool.map(process_frame, frames_to_process)

    face_vectors = []

    for frame_count, face_locations, face_encodings in results:
        for face_encoding in face_encodings:
            face_vectors.append((frame_count, face_encoding))

    # Save the face vectors to a file
    vectors_file = os.path.join(output_dir, "face_vectors.txt")
    with open(vectors_file, "w") as f:
        for frame_count, face_encoding in face_vectors:
            f.write(f"{frame_count}," + ",".join(f"{v:.8f}" for v in face_encoding) + "\n")

    cap.release()
    print("Face vectors saved.")

def main():
    video_path = r"C:\Users\user\Desktop\사진\KakaoTalk_20240624_153034158.mp4"
    output_directory = r"C:\Users\user\Desktop\output_frames"

    freeze_support()

    process_video(video_path, output_directory)

if __name__ == '__main__':
    main()
