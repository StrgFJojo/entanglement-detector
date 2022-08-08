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


# jojo 1.0 incoming


def run(
    video="",
    show_livestream=True,
    save_livestream=False,
    save_output_table=False,
    save_camera_input=False,
    synch_metric="2pax_90",
    cpu=True,
    net="",
):
    print(f"Style is {synch_metric}")
    # Check arguments
    synch_styles_excl_int_params = (
        synchrony_detection.SynchronyDetector.synch_styles_excl_int_params
    )
    if video == "":
        raise ValueError("-video has to be provided")
    if net == "":
        raise ValueError("-checkpoint-path has to be provided")
    if synch_metric not in synch_styles_excl_int_params:
        try:
            int(synch_metric)
        except ValueError:
            print(
                f"{synch_metric}is not a valid input for argument synch_style"
            )
            exit(1)

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

    # Setup entanglement detection
    synch_detector = synchrony_detection.SynchronyDetector(synch_metric)
    distance_calculator = distance_calculation.DistanceCalculator()
    height_calculator = height_calculation.HeightCalculator()

    # Setup output generation
    delay = 1
    if save_livestream:
        output_handler_video = output_creation.OutputHandler(
            output_type="video", file_name="output_video.avi"
        )
    if save_camera_input:
        output_handler_video_raw = output_creation.OutputHandler(
            output_type="video", file_name="input_video.avi"
        )
    if save_output_table:
        output_handler_table = output_creation.OutputHandler(
            output_type="table", file_name="output_table.csv"
        )

    # Setup visualization
    visualizer = visualization.Visualizer()

    # Iterate over video frames
    frame_provider = input_handling.VideoReader(video)
    for frame_idx, img in enumerate(
        tqdm(frame_provider, desc="Frame processing")
    ):
        # for non-webcam input, only process every 30th frame
        if video != "0" and frame_idx % 30 != 0:
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
        relevant_poses = synch_detector.get_relevant_poses(all_poses)

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
            cv2.imshow("Synch Detector", visualizer.img)
            key = cv2.waitKey(delay)
            if key == 27:  # esc
                cv2.destroyAllWindows()
                for i in range(1, 5):
                    cv2.waitKey(1)
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
        "--synch-metric", default="2pax_90", help="synchrony metric to be used"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="run inference on cpu"
    )
    parser.add_argument("--checkpoint-path", default="", type=str)
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
    )

    exit()
