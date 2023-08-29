import streamlit as st
import cv2
import os
import re
plugin_info = {"name": "Video-to-Grid"}


def create_grid_from_video(video_path, nth_frame, grid_advance, frame_limit, grid_limit, target_resolution, output_resolution, num_rows, num_cols, output_dir="output"):
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

    source_frame_count = frame_count = grid_count = 0
    frame_distance = num_rows * num_cols
    frames = []

    print_log(f'''Processing every {nth_frame} frame{'s' if nth_frame > 1 else ''} into a {num_cols} x {num_rows} grid of {frame_distance} cells.  
Resolution: Cell {target_resolution[0]}x{target_resolution[1]}px | Output: {output_resolution[0]}x{output_resolution[1]}px  
Grid advance: {grid_advance} | Frame limit: {frame_limit}''')

    while True:
        # frame or grid limit reached
        frame_limited = frame_limit != -1 and frame_count >= frame_limit
        grid_limited = grid_limit != -1 and grid_count >= grid_limit

        # Advance source frame count, get frame, break loop if at a limit or finished
        ret, frame = cap.read()
        source_frame_count += 1
        if not ret or frame_limited or grid_limited:
            break

        # Only use every nth frame from source video
        if source_frame_count % nth_frame == 0:
            # Advance frame count for each frame resized and appended
            resized_frame = cv2.resize(frame, target_resolution)
            frames.append(resized_frame)
            frame_count += 1

            # If we have collected the desired number of frames, create the grid
            if len(frames) >= frame_distance:
                # Organize frames into rows and rows into grid
                rows = []
                for i in range(0, len(frames), num_cols):
                    row_frames = frames[i:i + num_cols]
                    rows.append(cv2.hconcat(row_frames))
                grid_image = cv2.vconcat(rows)

                # Advance grid count for each grid resized and saved
                resized_grid = cv2.resize(grid_image, output_resolution)
                output_path = os.path.join(output_dir, f"grid_{grid_count}.jpg")
                cv2.imwrite(output_path, resized_grid)
                grid_count += 1

                # remove first cell or all cells
                if grid_advance == "Cell":
                    frames.pop(0)
                else:
                    frames = []

    # Close the video capture
    cap.release()

    print_log(f"Processed and saved grid images in {output_dir}", color_string="rainbow")

def plugin_tab(*args, **kwargs):
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save the uploaded video to a temporary location
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        print_log('Video uploaded successfully!')

        cols = st.columns((2,1), gap="small")
        with cols[0]:
            row1 = st.columns((1,1), gap="small")
            with row1[0]:
                nth_frame = st.number_input("Extract every N frames", 1, 60, 1)
            with row1[1]:
                frame_limit = st.number_input("Frame maximum (-1 unlimited)", -1, None, -1)

            row2 = st.columns((1,1), gap="small")
            with row2[0]:
                num_cols = st.number_input("Number of columns", 1, 32, 1)
            with row2[1]:
                num_rows = st.number_input("Number of rows", 1, 32, 2)


            row3 = st.columns((1,1), gap="small")
            with row3[0]:
                width = st.number_input("Cell width", 64, 2048, 1024)
            with row3[1]:
                height = st.number_input("Cell height", 64, 2048, 512)

            row4 = st.columns((1,1), gap="small")
            with row4[0]:
                output_width = st.number_input("Output width", 64, 4096, 1024)
            with row4[1]:
                output_height = st.number_input("Output height", 64, 4096, 1024)

        with cols[1]:
            grid_limit = st.number_input("Grid maximum (-1 unlimited)", -1, None, 500)
            grid_advance = st.radio("Grid advance to next",
                                    ["Frame", "Cell"],
                                    captions = ["[1,2,3,4][5,6,7,8][9,10,11,12]", "[1,2,3,4][2,3,4,5][3,4,5,6]"],
                                    horizontal = True)

        # Get parameters from user
        output_dir = st.text_input("Output directory", "output")

        # set tuple from dimensions
        target_resolution = (width, height)
        output_resolution = (output_width, output_height)

        if st.button("Create Grids"):
            # Create grid images
            create_grid_from_video("temp_video.mp4", nth_frame, grid_advance, frame_limit, grid_limit, target_resolution, output_resolution, num_rows, num_cols, output_dir)

def print_log(msg, color_string=None):
    # print msg to console in raw form
    print(msg)

    # if color code provided, combine with msg for browser
    if color_string:
        msg = f":{color_string}[{msg}]"
    st.markdown(msg)
