from google.colab import drive
drive.mount('/content/drive',force_remount=True)

cd 'drive/MyDrive/CS 231N Project/YOLO/PyTorch-YOLOv3'

!pip install importlib-metadata==3.7.3
!pip install -q --pre poetry
!poetry --version

!poetry install

%load_ext tensorboard

%tensorboard --host localhost --logdir='logs' --port=6006

