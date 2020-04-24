
from torchvision import models, datasets, transforms
from PIL import Image
import time
import sys

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

import os

import pycuda.driver as cuda

'''def Net():
    print("Loading in pretrained network ...")
    model = models.resnet101(pretrained=True)    # Get pretrained network
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    print("Number of inputs to final fully connected layer: %d" % num_features)

    # Change final layer.
    # Parameters of newly constructed modules have requires_grad=True by default.
    num_classes = 30
    model.fc = nn.conv1 = nn.Conv2d(num_features, num_classes, kernel_size=3, padding=1)
    return model'''
def Net():
    print("Loading in pretrained network ...")
    model = models.segmentation.fcn_resnet101(pretrained=True).eval().cuda()
    for param in model.parameters():
        param.requires_grad = False

    '''num_features = model.fc.in_features
    print("Number of inputs to final fully connected layer: %d" % num_features)

    # Change final layer.
    # Parameters of newly constructed modules have requires_grad=True by default.
    num_classes = 30
    model.fc = nn.conv1 = nn.Conv2d(num_features, num_classes, kernel_size=3, padding=1)'''
    model.aux_classifier = nn.Sequential(
    nn.Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.Dropout(p=0.2, inplace=False),
    nn.Conv2d(256, 30, kernel_size=(1, 1), stride=(1, 1))
    )
    print(model)
    return model

class Cityscapes(Dataset):
    def __init__(self, **kwargs):
        self.dataset = datasets.Cityscapes(**kwargs)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, item):
        the_data, label = self.dataset[item]
        return(the_data, transforms.ToTensor()(label))

def train_model(model, train_loader, val_loader, epochs=1):
    learning_rate = 1e-4
    optimizer = optim.Adam(model.aux_classifier.parameters(), lr=learning_rate)

    entropyLoss = nn.CrossEntropyLoss()
    model = model.cuda()  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print("Epoch: ", e)
        for t, (x, y) in enumerate(train_loader):
            model.train()  # put model to training mode
            x = x.cuda()
            y = y.type(torch.LongTensor)
            y = y.cuda()

            scores = model(x)['out']
            print(x.shape,y.shape,scores.shape)
            loss = entropyLoss(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % 100 == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_val_accuracy(val_loader, model)
                print()


# Define the helper function
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

def semantic_segmentation(model, start_time, name):

    video = cv2.VideoCapture("VID_20200422_173227.mp4")
    is_ok, bgr_image_input = video.read()
    #print(bgr_image_input.shape)
    frameNo = 0

    if not is_ok:
        print("Cannot read video source")
        sys.exit()

    h1 = bgr_image_input.shape[0]
    w1 = bgr_image_input.shape[1]
    ##print(h1,w1)

    try:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fname = "OUTPUT2.avi"
        fps = 30.0
        videoWriter = cv2.VideoWriter(fname, fourcc, fps, (341, 256))
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
        input_image = input_image.cuda()

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
        cv2.imwrite(name + "_" + str(frameNo) + ".jpg", bgr_image)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

    videoWriter.release()
    return 0

'''def semantic_segmentation_cityscapes(cityscapes, path):
    video = cv2.VideoCapture(path)
    is_ok, bgr_image_input = video.read()
    print(bgr_image_input.shape)
    frameNo = 0

    if not is_ok:
        print("Cannot read video source")
        sys.exit()

    h1 = bgr_image_input.shape[0]
    w1 = bgr_image_input.shape[1]

    try:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fname = "OUTPUT1.avi"
        fps = 30.0
        videoWriter = cv2.VideoWriter(fname, fourcc, fps, (455, 256))
    except:
        print("Error: can't create output video: %s" % fname)
        sys.exit()

    while True:
        frameNo = frameNo + 1


        val_loader = torch.utils.data.DataLoader(dataset='./Cityscapes',
                                                 batch_size=2, shuffle=False,
                                                 num_workers=1)

        unsorted_img_ids = []
        print("11111111111")
        for step, (imgs, img_ids) in enumerate(val_loader):
            print("2222222222")
            with torch.no_grad():  # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
                print("3333333333333")
                imgs = Variable(imgs).cuda()  # (shape: (batch_size, 3, img_h, img_w))

                outputs = cityscapes(imgs)  # (shape: (batch_size, num_classes, img_h, img_w))

                outputs = outputs.data.cpu().numpy()  # (shape: (batch_size, num_classes, img_h, img_w))
                pred_label_imgs = np.argmax(outputs, axis=1)  # (shape: (batch_size, img_h, img_w))
                pred_label_imgs = pred_label_imgs.astype(np.uint8)

                for i in range(pred_label_imgs.shape[0]):
                    pred_label_img = pred_label_imgs[i]  # (shape: (img_h, img_w))
                    img_id = img_ids[i]
                    img = imgs[i]  # (shape: (3, img_h, img_w))

                    img = img.data.cpu().numpy()
                    img = np.transpose(img, (1, 2, 0))  # (shape: (img_h, img_w, 3))
                    img = img * np.array([0.229, 0.224, 0.225])
                    img = img + np.array([0.485, 0.456, 0.406])
                    img = img * 255.0
                    img = img.astype(np.uint8)

                    pred_label_img_color = label_img_to_color(pred_label_img)
                    overlayed_img = 0.35 * img + 0.65 * pred_label_img_color
                    overlayed_img = overlayed_img.astype(np.uint8)
                    cv2.imshow("VID", overlayed_img)

                    key_pressed = cv2.waitKey(1) & 0xFF
                    if key_pressed == 27 or key_pressed == ord('q'):
                        break

            cv2.imwrite("cityscapes.jpg",overlayed_img)
            out = cv2.VideoWriter("%s/stuttgart_%s_combined.avi" % (cityscapes.model_dir, sequence),
                                  cv2.VideoWriter_fourcc(*"MJPG"), 20, (2 * img_w, 2 * img_h))
            sorted_img_ids = sorted(unsorted_img_ids)
            for img_id in sorted_img_ids:
                img = cv2.imread(cityscapes.model_dir + "/" + img_id + ".png", -1)
                pred_img = cv2.imread(cityscapes.model_dir + "/" + img_id + "_pred.png", -1)
                overlayed_img = cv2.imread(cityscapes.model_dir + "/" + img_id + "_overlayed.png", -1)
    
                combined_img = np.zeros((2 * img_h, 2 * img_w, 3), dtype=np.uint8)
    
                combined_img[0:img_h, 0:img_w] = img
                combined_img[0:img_h, img_w:(2 * img_w)] = pred_img
                combined_img[img_h:(2 * img_h), (int(img_w / 2)):(img_w + int(img_w / 2))] = overlayed_img
    
                out.write(combined_img)
    
            out.release()'''

def semantic_segmentation_image(model, path, start_time, name):

    bgr_image_input = cv2.imread(path)
    transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_image = transform(Image.fromarray(bgr_image_input)).unsqueeze(0)
    input_image = input_image.cuda()

    out = model(input_image)['out']
    # print(out.shape)

    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    # print(om.shape)

    rgb = decode_segmap(om)
    rgb = np.array(rgb)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    #print(bgr.shape)
    bgr_image = cv2.resize(bgr_image_input, (bgr.shape[1], bgr.shape[0]))
    bgr_image = cv2.addWeighted(bgr, 0.5, bgr_image, 0.5, 0)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    cv2.imwrite(name + "_" + path, bgr_image)
    cv2.imshow("Segmented Image", bgr_image)


def main():

    try:
        print("Loading FCN-Resnet101 Model")
        fcn = torch.load('fcn-resnet101.model')
        print("Loaded FCN-Resnet101 Model succesfully")
    except:
        print("Downloading FCN-Resnet101 Model")
        fcn = models.segmentation.fcn_resnet101(pretrained=True).eval().cuda()
        torch.save(fcn, 'fcn-resnet101.model')

    try:
        print("Loading Cityscapes Dataset")
        cityscapes = torch.load('cityscapes_dataset')
        print("Loaded Cityscapes Dataset succesfully")
    except:
        print("Downloading Cityscapes Dataset")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.9,1.0), ratio=(0.9,1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        cityscapes = datasets.Cityscapes('./Cityscapes/', split='train_extra', mode='coarse', target_type='semantic', transform = transform)
        torch.save(cityscapes, 'cityscapes_dataset')


    try:    
        print("Loading DeepLabV3-Resnet101 Model")
        deeplab = torch.load('deeplabv3-resnet101.model')
        print("Loading DeepLabV3-Resnet101 Model succesfully")
    except:
        print("Downloading DeepLabV3-Resnet101 Model")
        deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval().cuda()
        torch.save(deeplab, 'deeplabv3-resnet101.model')
    

    start_time = time.time()
    semantic_segmentation(deeplab, start_time, "DeepLabV3")
    #semantic_segmentation_image(deeplab, 'police.jpg', start_time, "DeepLabV3")
    #semantic_segmentation_cityscapes(cityscapes, 'police.jpg')
    '''try:
        model = torch.load('resnet101_cityscapes.model')
    except:
        model = Net()
        model = model.cuda()
        train_transform = transforms.Compose([
            #transforms.RandomResizedCrop(size=224, scale=(0.9,1.0), ratio=(0.9,1.1)),
            #transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        test_transform = transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        print("HHH")
        #cityscapes = datasets.Cityscapes('./Cityscapes/', split='train_extra', mode='coarse', target_type='semantic', transform = transform)
        train_dataset = Cityscapes(root = "./Cityscapes", split="train", transform=train_transform)
        num_images = len(train_val_dataset)
        num_train = int(0.6 * num_images)
        num_val = num_images - num_train
        #num_test = num_images - num_train - num_val
        print("A")
        val_dataset = Cityscapes(root = "./Cityscapes", split="val", transform=train_transform)
        print("Number of training images: %d" % len(train_dataset))
        print("Number of validation images: %d" % len(val_dataset))

        print("B")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=1)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=3, shuffle=True, num_workers=1)
        print("C")
        #test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=True, num_workers=4)
        test_dataset = Cityscapes(root = "./Cityscapes", split="test", transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=1)
        print("Number of test images: %d" % len(test_dataset))
        print("D")
        torch.save(train_loader,'train_loader')
        torch.save(val_loader,'val_loader')
        torch.save(test_loader,'test_loader')
        train_model(model, train_loader, val_loader, epochs=3)
        print("E")
        torch.save(model, 'resnet101_cityscapes.model')'''

    #cv2.waitKey(0)


if __name__ == "__main__":
    main()
