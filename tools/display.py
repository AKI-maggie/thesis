# This file contains tool functions that are used for image display
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from tools.measurements import check_IoU

# functions used for display some predicting results


# label different segmentation parts with different colors
def give_color_to_seg_img(seg,n_classes):
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)
    
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)        

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


# output the original images, the predictions, and the label images
def show_results(test_data, test_predicts, test_labels, n_classes):
    # store all ious for late miou
    ious = []
    
    for i in range(len(test_data)):
        img  = test_data[i]
        seg = test_predicts[i]
        segtest = test_labels[i]

        fig = plt.figure(figsize=(20,60))    
        ax = fig.add_subplot(1,6,1)
        ax.imshow(img)
        ax.set_title("original")
        
        print(seg.shape)

        ax = fig.add_subplot(1,6,2)
        ax.imshow(give_color_to_seg_img(seg,n_classes))
        ax.set_title("predicted class")

        ax = fig.add_subplot(1,6,3)
        ax.imshow(rgb2gray(give_color_to_seg_img(seg,n_classes)), cmap="Reds_r")
        ax.set_title("predicted class grey-scale")

        ax = fig.add_subplot(1,6,4)
        ax.imshow(give_color_to_seg_img(segtest,n_classes))
        ax.set_title("true class")

        ax = fig.add_subplot(1,6,5)
        ax.imshow(rgb2gray(give_color_to_seg_img(segtest,n_classes)), cmap="Blues_r")
        ax.set_title("true class grey-scale")
        
        ax = fig.add_subplot(1,6,6)
        ax.imshow(rgb2gray(give_color_to_seg_img(seg,n_classes)), cmap="Reds_r", alpha=0.7)
        ax.imshow(rgb2gray(give_color_to_seg_img(segtest,n_classes)), cmap="Blues_r", alpha=0.5)
        ax.set_title("IoU class")
        
        plt.show()
        
        
        iou = check_IoU(seg, segtest)
        print("IoU result: %.2f%%" % iou)
        
        ious.append(iou)
    
    # calculate mIoU
    miou = sum(ious)/len(ious)
    print("\nmIoU: %.2f%%\n"%miou)
        