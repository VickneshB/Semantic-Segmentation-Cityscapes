# Semantic-Segmentation-Pytorch
1.  Create an virtual environment using
	```
	$ python3 -m venv  virtual environment name.
	```
2.  Run 
	```
	$ pip3 install -r requirements.txt
	```
3.  Run either of the below of your choice.
	```
	$ python3 main.py -m model_name -p path_to_video_if_any
	```
for example,

(a)  Run
	```
	$ python3 main.py -m fcn-resnet101
	```
to run the FCN-Resnet101 model for live video, 

or run
     	```
     	$ python3 main.py -m fcn-resnet101 -p path_to_video
	```
to run the FCN-Resnet101 model for a recorded video.

(b)  Run
	```
	$ python3 main.py -m deeplabv3-resnet101
	```
to run the DeepLabV3-Resnet101 model for live video

or run
	```
     	$ python3 main.py -m deeplabv3-resnet101 -p path_to_video
	```
to run the DeepLabV3-Resnet101 model for a recorded video.


Click below for VIDEO:

[![Watch the video](https://img.youtube.com/vi/Zh-Ws4peAs0/sddefault.jpg)](https://youtu.be/Zh-Ws4peAs0)
