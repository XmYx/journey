import streamlit as st
import cv2
import os
plugin_info = {"name": "Video-to-Grid"}


def create_grid_from_video(video_path, frame_distance, target_resolution, num_rows, num_cols, output_dir="output"):
    """
    Creates grid images from pairs of video frames.

    :param video_path: Path to the video file.
    :param frame_distance: Distance between the frames to create the grid.
    :param target_resolution: Desired resolution for each frame before assembling into a grid.
    :param output_dir: Directory to save the grid images.
    """

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to the target resolution
        resized_frame = cv2.resize(frame, target_resolution)
        frames.append(resized_frame)
        frame_count += 1

        # If we have collected the desired number of frames, create the grid
        if frame_count % frame_distance == 0:

            # Organize frames into rows
            rows = []
            for i in range(0, len(frames), num_cols):
                row_frames = frames[i:i + num_cols]
                rows.append(cv2.hconcat(row_frames))

            # Combine rows to create the final grid image
            grid_image = cv2.vconcat(rows)

            # Save the grid image
            output_path = os.path.join(output_dir, f"grid_{frame_count // frame_distance}.jpg")
            cv2.imwrite(output_path, grid_image)

            # Reset frames for the next grid
            frames = []

    # Close the video capture
    cap.release()

    print(f"Processed and saved grid images in {output_dir}")


def plugin_tab(*args, **kwargs):
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save the uploaded video to a temporary location
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        st.write('Video uploaded successfully!')

        # Get parameters from user
        frame_distance = st.slider("Frame distance", 1, 10, 2)
        width = st.slider("Target width for each frame", 100, 1920, 1024)
        height = st.slider("Target height for each frame", 100, 1080, 512)
        target_resolution = (width, height)
        output_dir = st.text_input("Output directory", "output")

        # Get additional parameters from user
        num_cols = st.slider("Number of columns", 1, 10, 2)
        num_rows = st.slider("Number of rows", 1, 10, 2)

        if st.button("Create Grids"):
            # Create grid images
            create_grid_from_video("temp_video.mp4", frame_distance, target_resolution, num_rows, num_cols, output_dir)
            st.write(f"Processed and saved grid images in {output_dir}")