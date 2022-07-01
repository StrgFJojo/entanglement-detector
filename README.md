# Entanglement Detector

## Prerequisites
1. Install the requirements 
~~~
pip install -r requirements.txt
~~~
2. Download the pre-trained pose estimation model from [OpenPose Lightweight](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch):

    [Download link](https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth)


## Instructions 
First, get your models:

Download weights and model from the links below and add them to entanglement_detector/models.


Now, this is how you run this project from your console:
1. Go to project folder
2. activate venv with "source venv/bin/activate"
3. run "python -m detector.main --video 0"