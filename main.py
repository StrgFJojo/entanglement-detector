import argparse
import warnings

import cv2
import torch
from tqdm import tqdm

from detector import (
    distance_calculation,
    height_calculation,
    input_handling,
    output_creation,
    pose_estimation,
    synchrony_detection,
    visualization,
)
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state

warnings.filterwarnings("ignore")


def run(
    video="",
    show_livestream=True,
    save_livestream=False,
    save_output_table=False,
    save_camera_input=False,
    synch_metric="pss",
    cpu=True,
    net="",
    group_size="all",
    frame_skip=1,
):

    # Check arguments
    if video == "":
        raise ValueError("--video has to be provided")
    if net == "":
        raise ValueError("--checkpoint-path has to be provided")
    if synch_metric not in ["pss", "pos", "lss", "los"]:
        raise ValueError(
            f"{synch_metric} not a valid argument for synch_metric"
        )
    if frame_skip < 1 or not isinstance(frame_skip, int):
        raise ValueError("--frame-skip needs to be a positive integer")
    if group_size != "all" and group_size < 2:
        raise ValueError("--group-size needs to be an integer >= 2 or 'all'")

    print(
        f"System setup starting! "
        f"Synch metric: {synch_metric}, "
        f"Group size: {group_size}"
    )

    # Setup input handling
    frame_provider = input_handling.VideoReader(video)

    # Setup pose estimation
    height_size = 256
    stride = 8
    upsample_ratio = 4
    if not cpu and torch.cuda.is_available():
        net = net.cuda()
    previous_poses = []
    track = 1
    smooth = 1
    pose_estimator = pose_estimation.PoseEstimator(
        net, height_size, stride, upsample_ratio, cpu
    )

    # Setup synchronization detection
    synch_detector = synchrony_detection.SynchronyDetector(synch_metric)
    distance_calculator = distance_calculation.DistanceCalculator()
    height_calculator = height_calculation.HeightCalculator()

    # Setup output generation
    delay = 1
    if save_livestream:
        output_handler_video = output_creation.OutputHandler(
            output_type="video",
            file_name="output_video.avi",
            fps=frame_provider.fps,
        )
    if save_camera_input:
        output_handler_video_raw = output_creation.OutputHandler(
            output_type="video",
            file_name="input_video.avi",
            fps=frame_provider.fps,
        )
    if save_output_table:
        output_handler_table = output_creation.OutputHandler(
            output_type="table", file_name="output_table.csv", fps=None
        )

    # Setup visualization
    visualizer = visualization.Visualizer()
    print("Setup finished.")
    # Frame analysis
    print(
        "Starting frame analysis. "
        "To interrupt analysis, press 'esc' in the livestream window."
    )

    for frame_idx, img in enumerate(
        tqdm(
            frame_provider,
            desc="Frame processing",
            total=frame_provider.total_frames,
        )
    ):
        # For non-webcam input, skip frames if desired
        if video != "0" and frame_idx % frame_skip != 0:
            continue

        # Attach input frame to output video
        if save_camera_input:
            output_handler_video_raw.build_outputs(img)

        # Estimate poses
        all_poses = pose_estimator.img_to_poses(img)

        # Track poses between frames
        if track:
            pose_estimation.track_poses(
                previous_poses, all_poses, smooth=smooth
            )
            previous_poses = all_poses

        # Get poses relevant for entanglement detection
        relevant_poses = synch_detector.get_relevant_poses(
            all_poses, group_size
        )

        # Calculate body synchronization
        synchrony = synch_detector.calculate_synchrony(relevant_poses)

        # Calculate body distance
        distance_px = distance_calculator.calculate_distance(relevant_poses)
        heights = height_calculator.calculate_heights(relevant_poses)
        distance, normalized_distance = distance_calculator.normalize_distance(
            distance_px, heights
        )

        # Attach to output table
        if save_output_table:
            output_handler_table.build_outputs(
                {**synchrony, **normalized_distance}
            )

        # Generate video output with overlay
        if show_livestream or save_livestream:
            visualizer.setup(
                img,
                all_poses,
                relevant_poses,
                synchrony,
                synch_metric,
                distance,
            )

            # Draw bounding boxes
            visualizer.draw_bounding_boxes(track)

            # Add colored skeleton to indicate degree of synchronization
            visualizer.skeleton_overlay()

            # Add textbox with entanglement summary
            visualizer.text_overlay()

        # Display illustrated frame
        if show_livestream:
            cv2.imshow("Synch Detector", img)
            key = cv2.waitKey(delay)
            if key == 27:  # esc
                break
            elif key == 112:  # 'p'
                if delay == 1:
                    delay = 0
                else:
                    delay = 1

        # Attach illustrated frame to output video
        if save_livestream:
            output_handler_video.build_outputs(visualizer.img)

    # Iteration over frames finished (No frames left or keyboard interrupt)
    print("Frame analysis stopped. Closing files. Releasing outputs.")
    # Release resources
    if save_livestream:
        output_handler_video.release_outputs()
    if save_camera_input:
        output_handler_video_raw.release_outputs()
    if save_output_table:
        output_handler_table.release_outputs()
    if show_livestream:
        cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--video", type=str, default="", help="path to video file or camera id"
    )
    parser.add_argument(
        "--show-livestream", default=True, help="show detection on stream"
    )
    parser.add_argument(
        "--save-livestream", default=False, help="save illustrated input video"
    )
    parser.add_argument(
        "--save-output-table", default=False, help="save entanglement as csv"
    )
    parser.add_argument(
        "--save-camera-input", default=False, help="save input from camera"
    )
    parser.add_argument(
        "--synch-metric", default="pss", help="synchrony metric to be used"
    )
    parser.add_argument("--cpu", default=True, help="run inference on cpu")
    parser.add_argument("--checkpoint-path", default="", type=str)
    parser.add_argument(
        "--group-size",
        default="all",
        help="number of people for synch detection",
    )
    parser.add_argument(
        "--frame-skip", default=1, help="only every i-th frame gets analyzed"
    )

    args = parser.parse_args()

    # Setup pose estimation with OpenPose Lightweight
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    load_state(net, checkpoint)
    net = net.eval()

    run(
        args.video,
        args.show_livestream,
        args.save_livestream,
        args.save_output_table,
        args.save_camera_input,
        args.synch_metric,
        args.cpu,
        net,
        args.group_size,
        args.frame_skip,
    )

    exit()
