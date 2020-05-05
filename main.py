import sys
import numpy as np
import cv2
import os
import pycuda.driver as cuda
from torchvision import models, datasets, transforms
from PIL import Image
import time
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m","--model", help="model to be used", type=str)
parser.add_argument("-p","--path", help="path of the video", type=str)
args = parser.parse_args()
print("Model Chosen:", args.model)

if args.path == None:
    path = 0
else:
    path = args.path


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def semantic_segmentation(model, start_time, name, path = 0):

    video = cv2.VideoCapture(path)
    is_ok, bgr_image_input = video.read()

    frameNo = 0

    if not is_ok:
        print("Cannot read video source")
        sys.exit()

    h1 = bgr_image_input.shape[0]
    w1 = bgr_image_input.shape[1]
    ##print(h1,w1)

    try:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fname = "Segmented.avi"
        fps = 30.0
        videoWriter = cv2.VideoWriter(fname, fourcc, fps, (455,256))
    except:
        print("Error: can't create output video: %s" % fname)
        sys.exit()

    while True:
        frameNo = frameNo + 1

        is_ok, bgr_image_input = video.read()
        if not is_ok:
            break

        transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_image = transform(Image.fromarray(bgr_image_input)).unsqueeze(0)
        input_image = input_image.to(device)

        out = model(input_image)['out']
        #print(out.shape)

        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        #print(om.shape)

        rgb = decode_segmap(om)
        rgb = np.array(rgb)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        print(bgr.shape)
        bgr_image = cv2.resize(bgr_image_input, (bgr.shape[1], bgr.shape[0]))
        bgr_image = cv2.addWeighted(bgr, 0.5, bgr_image, 0.5, 0)
        elapsed_time = time.time() - start_time
        print(frameNo, elapsed_time)
        videoWriter.write(bgr_image)

        cv2.imshow("Segmented Image", bgr_image)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

    videoWriter.release()
    return 0


def main():
    
    if args.model == "fcn-resnet101":
        try:
            print("Loading FCN-Resnet101 Model")
            fcn = torch.load('fcn-resnet101.model')
            print("Loaded FCN-Resnet101 Model succesfully")
        except:
            print("Downloading FCN-Resnet101 Model")
            fcn = models.segmentation.fcn_resnet101(pretrained=True).eval().to(device)
            torch.save(fcn, 'fcn-resnet101.model')

        start_time = time.time()
        semantic_segmentation(fcn, start_time, "FCN", path)
    elif args.model == "deeplabv3-resnet101":
        try:    
            print("Loading DeepLabV3-Resnet101 Model")
            deeplab = torch.load('deeplabv3-resnet101.model')
            print("Loading DeepLabV3-Resnet101 Model succesfully")
        except:
            print("Downloading DeepLabV3-Resnet101 Model")
            deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval().to(device)
            torch.save(deeplab, 'deeplabv3-resnet101.model')

        start_time = time.time()
        semantic_segmentation(deeplab, start_time, "DeepLabV3", path)
    else:
        print("Error!!! Model should be either fcn-resnet101 or deeplabv3-resnet101")
        sys.exit()


if __name__ == "__main__":
    main()
