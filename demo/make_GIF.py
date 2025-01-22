import os
import cv2
from PIL import Image
import numpy as np

def create_smooth_side_by_side_video(sequence1, sequence2, output_path, frames_per_pair=100, fps=30):
    """
    Create a smooth side-by-side video from two sequences of images with interpolation.

    Args:
        sequence1 (list): List of file paths for the first sequence of images.
        sequence2 (list): List of file paths for the second sequence of images.
        output_path (str): Path to save the generated video.
        frames_per_pair (int): Number of frames for each pair of images to simulate video.
        fps (int): Frames per second for the video.
    """
    # Load the image sequences
    images1 = [Image.open(img) for img in sequence1]
    images2 = [Image.open(img) for img in sequence2]

    # Ensure both sequences have the same number of frames
    if len(images1) != len(images2):
        raise ValueError("Both sequences must have the same number of images.")

    # Ensure all images are the same size by resizing
    max_height = max(max(img.height for img in images1), max(img.height for img in images2))
    for i in range(len(images1)):
        if images1[i].height != max_height:
            images1[i] = images1[i].resize((images1[i].width, max_height), Image.ANTIALIAS)
        if images2[i].height != max_height:
            images2[i] = images2[i].resize((images2[i].width, max_height), Image.ANTIALIAS)

    # Function to smoothly blend between two images
    def blend_images(img1, img2, steps):
        frames = []
        for step in range(steps):
            blended = Image.blend(img1, img2, alpha=step / (steps - 1))  # Interpolate
            frames.append(blended)
        return frames

    # Prepare video writer
    width = images1[0].width + images2[0].width  # Side by side
    height = max_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create smooth frame sequences and write to video
    for img1, img2 in zip(images1, images2):
        # Get the blended frames for the current pair of images
        blended_frames = blend_images(img1, img2, frames_per_pair)
        
        # Combine images side by side and write each frame to the video
        for frame in blended_frames:
            combined_img = Image.new("RGB", (width, height))
            combined_img.paste(frame, (0, 0))
            combined_img.paste(frame, (img1.width, 0))

            # Convert the image to a format that OpenCV can handle (BGR)
            frame_bgr = np.array(combined_img)[:, :, ::-1]  # Convert RGB to BGR
            out.write(frame_bgr)

    out.release()
    print(f"Video saved to {output_path}")

# Example usage
#sequence1 = ["image1_1.png", "image1_2.png", "image1_3.png"]  # Replace with your first image sequence
#sequence2 = ["image2_1.png", "image2_2.png", "image2_3.png"]  # Replace with your second image sequence
#create_smooth_side_by_side_video(sequence1, sequence2, "smooth_side_by_side_video.mp4")

# sequence is all files in a given directory
sequence1_path = '/home/boux/code/str_tensor_obb/demo/rotated_outputs/img4'
sequence2_path = '/home/boux/code/str_tensor_obb/demo/rotated_outputs/img7'

sequence1 = [os.path.join(sequence1_path, img) for img in os.listdir(sequence1_path)]
sequence2 = [os.path.join(sequence2_path, img) for img in os.listdir(sequence2_path)]

create_smooth_side_by_side_video(sequence1, sequence2, "smooth_side_by_side_video.mp4")
