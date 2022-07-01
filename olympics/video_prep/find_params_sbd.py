# attention:
# the function we test for now works with the ContentDetector, not with ThresholdDetector
# if you want to reuse this test, please replace lines 12 onwards in shot_transition_detection.py with:
"""def find_scenes(video_path, threshold=12, min_scene_len=15,
                                                   fade_bias=0.0, add_final_scene=False, block_size=8):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector())
"""
# furthermore, add ThresholdDetector to imports: (in shot_transition_detection.py)
# from scenedetect import VideoManager, ThresholdDetector, AdaptiveDetector

# for test video beijing_tune_cut_params.mp4
# Cuts to be detected at seconds: 13-14, 18-19, 23
# should return threshold val at which all three cuts - soft and hard cuts - were found
import shot_transition_detection

video = '/Users/josephinevandelden/PycharmProjects/entanglement_video_prep/beijing_tune_cut_params.mp4'

for t in range(255):
    scenes = shot_transition_detection.find_scenes(video, threshold=t, min_scene_len=15,
                                                   fade_bias=0.0, add_final_scene=False, block_size=8)
    print("\nThreshold: %d, # Scenes identified: %d\n" % (t, len(scenes)))
    print(scenes)
    if (len(scenes) == 4) and (13 * 30 <= scenes[1][0] <= 14 * 30) and (18 * 30 <= scenes[1][0] <= 19 * 30) and (
            23 * 30 <= scenes[1][0] <= 24 * 30):
        print("threshold val: %d" % t)
        print(scenes)
