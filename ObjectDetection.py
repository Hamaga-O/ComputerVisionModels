import os
from os.path import exists, join, basename, splitext

import random
import PIL
import torchvision
import cv2
import numpy as np
import torch
import wget
torch.set_grad_enabled(False)
  
import time
import matplotlib
import matplotlib.pylab as plt
from vid import draw_mask 
from calcPCA import mainfunc

#Establish model with no gradient analysis as training is not performed
plt.rcParams["axes.grid"] = False
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model = model.eval()

with torch.no_grad():
    
    #establish video feed using opencv and create variables for colors and labels of bounding box and mask
    vid = cv2.VideoCapture(0)
    coco_names = ['unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in coco_names]
    #colors = zip(coco_names,colors)
    print ('color shape:', colors[0][0])
    color = random.choice(colors)
    
    #Create loop which applies the model to each frame of the video feed and applies the bounding box and mask
    while True:
        ret,frame = vid.read()
    
        frame = cv2.resize(frame, (680, 480), fx = 0, fy = 0,
                              interpolation = cv2.INTER_CUBIC)
        if not ret: 
            break
    
    #Run model on frame to obtain mask rcnn data which is stored in 'output' variable and time taken to execute 
        t = time.time()
        frame_tensor = torchvision.transforms.functional.to_tensor(frame)
        output = model([frame_tensor])[0]
        print('executed in %.3fs' % (time.time() - t))
        
        result_frame = np.array(frame.copy())
        for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
          if score > 0.7:
            
            # draw bounding box on the frame along with label and confidence score
            tl = round(0.002 * max(result_frame.shape[0:2])) + 1  # line thickness
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(result_frame, c1, c2, color, thickness=tl)
            # draw text
            display_txt = "%s: %.1f%%" % (coco_names[label], 100*score)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(display_txt, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(result_frame, c1, c2, color, -1)  # filled
            cv2.putText(result_frame, display_txt, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            
            
        #converting masks into binary mask based on score to display only identified objects
        masks = None
        for score, mask in zip(output['scores'], output['masks']):
          if score > 0.6:
            if masks is None:
              masks = mask
            else:
              masks = torch.max(masks, mask)
     
        masked_frame = draw_mask(mainfunc(result_frame),masks,colors)
        
        cv2.namedWindow("video")
        cv2.imshow("video",masked_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
