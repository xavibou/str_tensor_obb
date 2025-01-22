import cv2
import numpy as np
import os

def generate_rotated_frames(input_image_path, output_folder, num_frames=100, rotation_range=45, shift_range=50):
    """
    Generate a set of frames by gradually rotating and shifting an image.
    
    Args:
        input_image_path (str): Path to the input image.
        output_folder (str): Folder to save the generated frames.
        num_frames (int): Number of frames to generate.
        rotation_range (int): Maximum rotation angle in degrees (both directions).
        shift_range (int): Maximum shift in pixels (both x and y directions).
    """
    # Load the input image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Failed to load image from {input_image_path}")
        return
    
    # Get image dimensions
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    for i in range(num_frames):
        # Calculate rotation angle and shifts
        angle = np.sin(2 * np.pi * i / num_frames) * rotation_range
        dx = np.sin(2 * np.pi * i / num_frames) * shift_range
        dy = np.cos(2 * np.pi * i / num_frames) * shift_range
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotation_matrix[0, 2] += dx
        rotation_matrix[1, 2] += dy
        
        # Apply rotation and shift
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
        
        # Save the frame
        frame_path = os.path.join(output_folder, f"frame_{i:03d}.png")
        cv2.imwrite(frame_path, rotated_image)
        print(f"Saved frame {frame_path}")

if __name__ == "__main__":
    input_image_path = "/home/boux/code/str_tensor_obb/data/DOTA/split_ss_dota1_0/test/images/P1163__1024__5273___1648.png"
    output_folder = "/home/boux/code/str_tensor_obb/demo/rotated_inputs/img10"
    generate_rotated_frames(input_image_path, output_folder)
