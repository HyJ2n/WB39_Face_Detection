import cv2
import face_recognition
from multiprocessing import Pool, cpu_count, freeze_support
import os


# Global variable to check if 'q' key is pressed
stop_processing = False

def process_frame(frame_path):
    frame_count, frame = frame_path
    # Convert frame to RGB (required by face_recognition library)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    
    # Draw rectangles around detected faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    
    return frame, face_locations

def process_video(video_path, output_dir):
    global stop_processing

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Frame Size: {frame_width}x{frame_height}")

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total Frames: {total_frames}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare a list of frames to process
    frames_to_process = [(i, cap.read()[1]) for i in range(total_frames)]

    # Create a pool of processes
    with Pool(cpu_count()) as pool:
        # Process each frame using multiprocessing
        results = pool.map(process_frame, frames_to_process)

    # Save the processed frames with rectangles drawn around faces
    for idx, (processed_frame, face_locations) in enumerate(results):
        output_path = os.path.join(output_dir, f"frame_{idx + 1}.jpg")
        cv2.imwrite(output_path, processed_frame)

        # Debugging: Print number of face encodings detected in the current frame
        num_faces = len(face_locations)
        print(f"Frame {idx + 1} - Detected {num_faces} faces. Saved to {output_path}")

    # Release the video capture object
    cap.release()

    print("Processing complete.")

def main():
    # Example usage: Replace 'video_path' with the path to your video file
    video_path = r"C:\Users\user\Desktop\사진\KakaoTalk_20240624_153034158.mp4"
    output_directory = r"C:\Users\user\Desktop\hyojin"

    # Ensure this module is being run as the main process on Windows
    freeze_support()

    # Process the video and save frames with rectangles drawn around faces
    process_video(video_path, output_directory)

if __name__ == '__main__':
    main()
