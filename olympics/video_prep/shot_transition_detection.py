# Manual: https://pyscenedetect.readthedocs.io/projects/Manual/en/latest/api.html
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect import frame_timecode
from scenedetect.detectors import ContentDetector
import pandas as pd


def find_scenes(video_path, threshold):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list()


def time_convert(x):
    h, m, s = map(int, x.split(':'))
    return (h * 60 + m) * 60 + s


# check if timespan a and b overlap
def has_timely_overlap(start_a, end_a, start_b, end_b):
    return (start_a <= end_b) and (end_a >= start_b)


# check if a lies within b
def lies_within_span(start_a, end_a, start_b, end_b):
    return (start_b <= start_a) and (end_b >= end_a)


def get_scenes_within_timestamps(scenes, timestamps):
    # get fps
    fps = scenes[0][0].get_framerate()

    # create FrameTimecode object from timestamps 0_resources frame
    selected_frames = []
    for i in range(len(timestamps)):
        selected_frames.append((frame_timecode.FrameTimecode(timecode=timestamps.ts_perf_start[i], fps=fps),
                                frame_timecode.FrameTimecode(timecode=timestamps.ts_perf_end[i], fps=fps)))

    scenes_within_timestamps = []
    scenes_teams = []
    for i in range(len(scenes)):
        for j in range(len(selected_frames)):
            if lies_within_span(scenes[i][0], scenes[i][1], selected_frames[j][0], selected_frames[j][1]):
                scenes_within_timestamps.append((scenes[i][0], scenes[i][1]))
                scenes_teams.append([len(scenes_within_timestamps)-1,
                                     scenes[i][0], scenes[i][1],
                                     scenes[i][0].get_frames(), scenes[i][1].get_frames(),
                                     timestamps.team[j]])
    scenes_teams = pd.DataFrame(scenes_teams, columns=['scene_id',
                                                       'ts_start', 'ts_end',
                                                       'frame_start', 'frame_end',
                                                       'team'])

    return scenes_within_timestamps, scenes_teams
