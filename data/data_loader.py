import numpy as np
import os
import random
from cv2 import imread, resize
import scipy.io as io

# A prototype class which mentions basic data loading functions
class DataLoader():
    def __init__(self, img_path, label_path):
        self.img_p = img_path
        self.label_p = label_path
        self.img_ids = []
        
        # use a live loading for training samples later,
        # so the loader class would only save the ids of the 
        # training data
        if self._load() <= 0:
            print("No training data found.")
            os.quit()

    def _load(self):
        pass

# a data-loading class for SiftFlow data
class SiftFlowLoader(DataLoader):
    def __init__(self, img_path, label_path, test_num = 200):
        super().__init__(img_path, label_path)
        self.test_num = test_num
        self.class_num = 33
        
    def _load(self):
        for root, dirs, files in os.walk(self.img_p):
            for f in files:
                if f.endswith('.jpg'):
                    fid = f.split('.')[0]
                    self.img_ids.append(fid)
        return len(self.img_ids)

    # rule: load enough images that could provide k shots for each class
    def check_datatset_completence(self, counts, k):
        count = 0
        for each in counts:
            if each < k:
                return False
            count += 1
        return True

    def separate_labels(self, lb, nc):
        seg_lb = np.zeros((lb.shape[0], lb.shape[1], nc + 1))
        for i in range(nc):
            seg_lb[:,:,i] = (lb==i).astype(int)
        return seg_lb

    def generate_training_batches(self,  
                                  k = 1,   # k - number of shots for each class
                                  img_height = 256,
                                  img_width = 256):
        while True:
            # randomly choose a batch of k images
            yield (self._generate_train(k, img_height, img_width))

    def generate_testing_dataset(self, num = 10, 
                                 img_height = 256,
                                 img_width = 256):
        img_ids = self.img_ids[-self.test_num:]
        x_test = []
        y_test = []

        for fid in random.sample(img_ids, num):
            flabel_path = fid + '.mat'
            flpath = os.path.join(self.label_p, flabel_path)
            flabel = resize(io.loadmat(flpath)['S'], (img_height, img_width))
            flabel = self.separate_labels(flabel, self.class_num)

            fimg_path = fid + '.jpg'
            fpath = os.path.join(self.img_p, fimg_path)
            fimg = resize(imread(fpath) / 255, (img_height, img_width))
            x_test.append(fimg)
            y_test.append(flabel)
        
        return np.array(x_test), np.array(y_test)

    def _generate_train(self, k, img_height, img_width): 
        # load images and corresponding labels
        img_ids = self.img_ids[:-self.test_num]
        images = []
        labels = []

        counts = np.zeros(self.class_num)

        for fid in random.sample(img_ids, len(img_ids)):
            if self.check_datatset_completence(counts, k):
                break
            # print('Loading image of {0}'.format(fid))
            
            # load label
            flabel_path = fid + '.mat'
            flpath = os.path.join(self.label_p, flabel_path)
            flabel = resize(io.loadmat(flpath)['S'], (img_height, img_width))
            flabel_classes = np.unique(flabel)

            flag = 0
            for i in range(self.class_num):
                if counts[i] < k and i in flabel_classes:
                    counts[i] += 1
                    
                    if flag == 0:
                        # load img
                        fimg_path = fid + '.jpg'
                        fpath = os.path.join(self.img_p, fimg_path)
                        fimg = resize(imread(fpath) / 255, (img_height, img_width))
                        images.append(fimg)
                        labels.append(self.separate_labels(flabel, self.class_num))
                        # labels.append(flabel)
                        flag = 1

        images = np.array(images)
        labels = np.array(labels)
        # print(images.shape)
        return images, labels
