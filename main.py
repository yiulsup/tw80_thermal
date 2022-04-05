import os
import colorsys
import random
import serial

import cv2
import sys
import time
import numpy as np
import logging
import threading
import signal
import serial
import time
import numpy as np
import binascii
import queue
import pandas as pd
from scipy import interpolate
import

main_queue = queue.Queue()
uart = serial.Serial("COM8", 115200)

sharpening_mask1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpening_mask2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
mexican_hat_1 = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
mexican_hat_2 = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])


def th():
    global uart
    global main_queue
    sending_1 = [0x02, 0x00, 0x04, 0x00, 0x01, 0x55, 0xaa, 0x03, 0xFA]
    sending_2 = [0x02, 0x00, 0x04, 0x01, 0x01, 0x00, 0x5, 0x03, 0x01]
    # sending_2 = [0x02, 0x00, 0x04, 0x01, 0x01, 0x00, 0x0a, 0x03, 0x0E]
    sending_3 = [0x02, 0x00, 0x04, 0x02, 0x01, 0x00, 0x01, 0x03, 0x06]
    sending_4 = [0x02, 0x00, 0x04, 0x02, 0x01, 0x01, 0x01, 0x03, 0x07]

    cnt = 0
    cnt1 = 0
    cnt2 = 0

    frame = np.zeros(4800)

    time.sleep(0.1)
    print("second command to fly")
    uart.write(sending_2)
    time.sleep(0.1)
    first = 1
    image_cnt = 0
    passFlag = np.zeros(6)
    start_frame = 0
    uart.write(sending_4)
    begin = 0
    check_cnt = 0

    uart.write(sending_1)
    while True:
        line = uart.read()
        cnt = cnt + 1
        if cnt >= 9:
            cnt = 0
            break
    uart.write(sending_4)

    while True:
        try:
            # global fvs # FileVideoStream
            line = uart.read()
            cnt1 = cnt1 + 1
            if begin == 0 and cnt1 == 1:
                rawDataHex = binascii.hexlify(line)
                rawDataDecimal = int(rawDataHex, 16)
                if rawDataDecimal == 2:
                    begin = 1
                else:
                    begin = 0
                    cnt1 = 0
                    continue
            if begin == 1 and cnt1 == 20:
                for i in range(0, 9600):
                    line = uart.read()
                    cnt1 = cnt1 + 1
                    rawDataHex = binascii.hexlify(line)
                    rawDataDecimal = int(rawDataHex, 16)
                    if first == 1:
                        dec_10 = rawDataDecimal * 256
                        first = 2
                    elif first == 2:
                        first = 1
                        dec = rawDataDecimal
                        frame[image_cnt] = dec + dec_10
                        image_cnt = image_cnt + 1

                    if image_cnt >= 4800:
                        image_cnt = 0
                        error = np.mean(frame)
                        if error > 7 and error < 8:
                            continue
                        main_queue.put(frame)

            if cnt1 == 2 and begin == 1:
                rawDataHex = binascii.hexlify(line)
                rawDataDecimal = int(rawDataHex, 16)
                if rawDataDecimal == 0x25:
                    begin = 1
                else:
                    begin = 0
                    cnt1 = 0
                    continue
            if cnt1 == 3 and begin == 1:
                rawDataHex = binascii.hexlify(line)
                rawDataDecimal = int(rawDataHex, 16)
                if rawDataDecimal == 0xA1:
                    begin = 1
                else:
                    begin = 0
                    cnt1 = 0
                    continue

            if cnt1 == 9638 and begin == 1:
                begin = 0
                cnt1 = 0
            else:
                continue

        except:
            continue

t = threading.Thread(target=th)
t.start()


YOLO_net = cv2.dnn.readNet("yolov2-tiny_200000ab.weights", "yolov2-tiny.cfg")

classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]

while True:
    """
    frame1 = main_queue.get()
    max = np.max(frame1)
    min = np.min(frame1)
    #print(np.mean(frame1))

    nfactor = 255 / (max - min)
    pTemp = frame1 - min
    nTemp = pTemp * nfactor
    frame1 = nTemp
    image = frame1.reshape(60, 80)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    uint_img = np.array(image).astype('uint8')
    #grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
    """
    uint_img = cv2.imread("prisonw276.png", cv2.IMREAD_COLOR)
    #uint_img = cv2.cvtColor(uint_img, cv2.COLOR_BGR2GRAY)
    uint_img = cv2.resize(uint_img, (0, 0), fx=6.0, fy=6.0, interpolation=cv2.INTER_CUBIC)

    #img_sobel_x = cv2.Sobel(uint_img, cv2.CV_64F, 1, 0, ksize=3)
    #img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

    #img_sobel_y = cv2.Sobel(uint_img, cv2.CV_64F, 0, 1, ksize=3)
    #img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

    #img_sobel = abs(img_sobel_x * 0.8) + abs(img_sobel_y * 0.6)

    #img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)
    #img_sobel = cv2.resize(img_sobel, (640, 480))

    cv2.imshow('d', uint_img)
    cv2.waitKey(1)
    continue
    #df = pd.DataFrame(uint_img)

    #df.to_csv('prison.csv', index=False)

    #sharp_img = cv2.filter2D(uint_img, -1, sharpening_mask1)
    #uint_img = cv2.Canny(sharp_img, 120, 140)
    #uint_img = cv2.resize(uint_img, (640, 480))
    #uint_img = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
    ret, img_binary = cv2.threshold(uint_img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(img_binary, [cnt], 0, (255, 0, 0), 3)
    grayImage = cv2.resize(contours, (160, 160))

    #cv2.imshow('d', sharp_img)
    #cv2.waitKey(1)

    #continue

    h, w, c = grayImage.shape
    blob = cv2.dnn.blobFromImage(grayImage, 0.00392, (160, 160), (0, 0, 0), True, crop=False)


    YOLO_net.setInput(blob)
    outs = YOLO_net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    x = 0
    y = 0
    dw = 0
    dh = 0
    label = "None"

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8:
                # Object detected
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.19, 0.4)

    if len(class_ids) != 0:
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                score = confidences[i]
                print("{} : {}, {}, {}, {}".format(label, x, y, dw, dh))

                # 경계상자와 클래스 정보 이미지에 입력
                grayImage = cv2.rectangle(grayImage, (x, y), (x + w, y + h), (0, 0, 255), 5)
                grayImage = cv2.putText(grayImage, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

    else:
        pass
        #errint1 = int(frame1[0])
        #errint2 = int(frame1[1])
        #errint3 = int(frame1[2])
        #if errint1 == errint2:
        #    continue
        #print("{} : {}, {}, {}, {}".format(label, x, y, dw, dh))

    frame1 = cv2.resize(grayImage, (640, 480))

    cv2.imshow("YOLOv3", frame1)
    cv2.waitKey(1)