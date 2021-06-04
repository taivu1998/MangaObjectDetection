from google.colab import drive
drive.mount('/content/drive',force_remount=True)

cd 'drive/MyDrive/CS 231N Project/YOLO/PyTorch-YOLOv3'

!pip install importlib-metadata==3.7.3
!pip install -q --pre poetry
!poetry --version

!poetry install

cat config/yolov3-awesomeconfig.cfg

!poetry run yolo-train --model config/yolov3-awesomeconfig.cfg --data config/custom.data --pretrained_weights weights/darknet53.conv.74