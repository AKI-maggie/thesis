# This file contains tool functions that are used for image display
import numpy as np
import seaborn as sns

# label different segmentation parts with different colors
def give_color_to_seg_img(seg,n_classes):
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes+1)
    
    for c in range(n_classes+1):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)        