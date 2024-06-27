import os

def extract_frame_range_from_filenames(folder):
    frame_ranges = {
        'Female': [],
        'Male': []
    }
    
    for gender_folder in ['Female', 'Male']:
        gender_folder_path = os.path.join(folder, gender_folder)
        if os.path.exists(gender_folder_path):
            filenames = os.listdir(gender_folder_path)
            if filenames:
                frame_numbers = [int(filename.split('_')[1]) for filename in filenames]
                min_frame = min(frame_numbers)
                max_frame = max(frame_numbers)
                frame_ranges[gender_folder] = [min_frame, max_frame]
    
    return frame_ranges

def main():
    output_folder = r'C:\Users\user\Desktop\wb39_yolov8\gender_model\face_output'
    
    frame_ranges = extract_frame_range_from_filenames(output_folder)
    
    print("Female frame ranges:", frame_ranges['Female'])
    print("Male frame ranges:", frame_ranges['Male'])

if __name__ == '__main__':
    main()
