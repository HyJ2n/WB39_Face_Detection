import cv2
import os

def extract_frames_and_save(input_video_path, output_video_path, start_frame, end_frame):
    # Open the input video file
    video_capture = cv2.VideoCapture(input_video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file '{input_video_path}'")
        return
    
    # Get frame width, height, and FPS
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Adjust codec as needed for your system
    out = cv2.VideoWriter(output_video_path + '.mp4', fourcc, fps, (frame_width, frame_height))
    
    # Set frame position to start_frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read and write frames from start_frame to end_frame
    frame_count = start_frame
    while frame_count <= end_frame:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        out.write(frame)
        
        # Display progress
        print(f"Writing frame {frame_count}")
        
        frame_count += 1
    
    # Release VideoCapture and VideoWriter
    video_capture.release()
    out.release()
    print(f"Extracted frames from {start_frame} to {end_frame} and saved as '{output_video_path}'")

def main():
    input_video_path = r"C:\Users\user\Desktop\사진\KakaoTalk_20240624_153034158.mp4"
    output_video_dir = r"C:\Users\user\Desktop\wb39_yolov8\gender_model\appear_video"
    output_video_name = "female"
    start_frame = 296
    end_frame = 416
    
    # Ensure output directory exists
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
    
    output_video_path = os.path.join(output_video_dir, output_video_name)
    
    extract_frames_and_save(input_video_path, output_video_path, start_frame, end_frame)

if __name__ == '__main__':
    main()
