import numpy as np
import os
import random
from cv2 import imread, resize
import scipy.io as io
from numpy.random import randint

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

class CamVidLoader(DataLoader):
    def __init__(self, img_path, label_path, test_num = 200):
        self.label_names = ['animal','archway','bicyclist','bridge','building','car','pram','child','column',
                            'fence','drive','lane','text','scooter','others','parking','pedestrian','road','shoulder',
                            'sidewalk','sign','sky','suv','traffic cone','traffic light','train','tree','truck',
                            'tunnel', 'vegetation','void','wall']
        super().__init__(img_path, label_path)
        self.test_num = test_num
        self.class_num = 32

        self.test_data = self.generate_testing_samples()

        # a data-loading class for SiftFlow data
        
    def _load(self):
        for root, dirs, files in os.walk(self.img_p):
            for f in sorted(files):
                if f.endswith('.png'):
                    fid = f.split('.')[0]
                    self.img_ids.append(fid)
        self.ids = np.arange(len(self.img_ids))
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
        seg_lb = np.zeros((lb.shape[0], lb.shape[1], nc))
        for i in range(nc):
            seg_lb[:,:,i] = (lb==i).astype(int)
        return seg_lb

    def generate_supervised_samples(self,  
                                  k = 1,   # k - number of shots for each class
                                  img_height = 256,
                                  img_width = 256):
        while True:   
            # randomly choose a batch of k images
            yield (self._generate_train(k, img_height, img_width))

    def generate_testing_samples(self, k = 1, 
                                 img_height = 256,
                                 img_width = 256):
        images = []
        labels = []

        counts = np.zeros(self.class_num)

        for fid in random.sample(self.img_ids, len(self.img_ids)):
            if self.check_datatset_completence(counts, k):
                break
            # print('Loading image of {0}'.format(fid))
            
            # load label
            flabel_path = fid + '_L.png'
            flpath = os.path.join(self.label_p, flabel_path)
            flabel = resize(imread(flpath), (img_height, img_width)).astype(int)[:,:,0]
            flabel_classes = np.unique(flabel)
            flabel = self.separate_labels(flabel, self.class_num)

            for i in range(self.class_num):
                if counts[i] < k and i in flabel_classes:
                    counts[i] += 1
                    
                    # load img
                    fimg_path = fid + '.png'
                    fpath = os.path.join(self.img_p, fimg_path)
                    fimg = resize(imread(fpath)/255, (img_height, img_width))
                    images.append(fimg)
                    labels.append(flabel)

        images = np.array(images)
        labels = np.array(labels)
        # print(images.shape)
        return images, labels

    def generate_real_samples(self, dataset, n_samples, seed = None):
        imgs, labels = dataset
        if seed:
            np.random.seed(seed)
        # choose random instances
        idx = randint(0, imgs.shape[0], n_samples)
        # load images and labels
        x, y = imgs[idx], labels[idx]
        # generate class labels
        y2 = np.ones([n_samples, x.shape[1], x.shape[2]])
        return [x, y], y2

    # generate labeled training dataset
    def _generate_train(self, k, img_height, img_width): 
        # load images and corresponding labels
        # ids = self.ids[:-self.test_num]
        images = []
        labels = []

        counts = np.zeros(self.class_num)

        for fid in random.sample(self.img_ids, len(self.img_ids)):
            if self.check_datatset_completence(counts, k):
                break
            # print('Loading image of {0}'.format(fid))
            
            # load label
            flabel_path = fid + '_L.png'
            flpath = os.path.join(self.label_p, flabel_path)
            flabel = resize(imread(flpath), (img_height, img_width)).astype(int)[:,:,0]
            flabel_classes = np.unique(flabel)
            flabel = self.separate_labels(flabel, self.class_num)

            for i in range(self.class_num):
                if counts[i] < k and i in flabel_classes:
                    counts[i] += 1
                
                    # load img
                    fimg_path = fid + '.png'
                    fpath = os.path.join(self.img_p, fimg_path)
                    fimg = resize(imread(fpath)/255, (img_height, img_width))
                    images.append(fimg)
                    labels.append(flabel)

        images = np.array(images)
        labels = np.array(labels)
        # print(images.shape)
        return images, labels


# a data-loading class for SiftFlow data
class SiftFlowLoader(DataLoader):
    def __init__(self, img_path, label_path, test_num = 200):
        super().__init__(img_path, label_path)
        self.test_num = test_num
        self.class_num = 34

        self.test_data = self.generate_testing_samples()
        
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
        seg_lb = np.zeros((lb.shape[0], lb.shape[1], nc))
        for i in range(nc):
            seg_lb[:,:,i] = (lb==i).astype(int)
        return seg_lb

    def generate_supervised_samples(self,  
                                  k = 1,   # k - number of shots for each class
                                  img_height = 256,
                                  img_width = 256):
        while True:
            # randomly choose a batch of k images
            yield (self._generate_train(k, img_height, img_width))

    def generate_testing_samples(self, k = 1, 
                                 img_height = 256,
                                 img_width = 256):
        # img_ids = self.img_ids[-self.test_num:]
        # load images and corresponding labels
        images = []
        labels = []

        counts = np.zeros(self.class_num)

        for fid in random.sample(self.img_ids, len(self.img_ids)):
            if self.check_datatset_completence(counts, k):
                break
            # print('Loading image of {0}'.format(fid))
            
            # load label
            flabel_path = fid + '.mat'
            flpath = os.path.join(self.label_p, flabel_path)
            flabel = resize(io.loadmat(flpath)['S'], (img_height, img_width))
            # fflabel = 
            flabel_classes = np.unique(flabel)

            flag = 0
            for i in range(self.class_num):
                if counts[i] < k and i in flabel_classes:
                    counts[i] += 1
                    flag = 1

            # load img
            if flag == 1:
                fimg_path = fid + '.jpg'
                fpath = os.path.join(self.img_p, fimg_path)
                fimg = resize(imread(fpath) / 255, (img_height, img_width))
                images.append(fimg)
                labels.append(self.separate_labels(flabel, self.class_num))
            # labels.append(flabel)

        images = np.array(images)
        labels = np.array(labels)
        # print(images.shape)
        return images, labels

    def generate_real_samples(self, dataset, n_samples, seed = None):
        imgs, labels = dataset
        if seed:
            np.random.seed(seed)
        # choose random instances
        idx = randint(0, imgs.shape[0], n_samples)
        # load images and labels
        x, y = imgs[idx], labels[idx]
        # generate class labels
        y2 = np.ones([n_samples, x.shape[1], x.shape[2]])
        return [x, y], y2

    # generate labeled training dataset
    def _generate_train(self, k, img_height, img_width): 
        # load images and corresponding labels
        # img_ids = self.img_ids[:-self.test_num]
        images = []
        labels = []

        counts = np.zeros(self.class_num)

        for fid in random.sample(self.img_ids, len(self.img_ids)):
            if self.check_datatset_completence(counts, k):
                break
            # print('Loading image of {0}'.format(fid))
            
            # load label
            flabel_path = fid + '.mat'
            flpath = os.path.join(self.label_p, flabel_path)
            flabel = resize(io.loadmat(flpath)['S'], (img_height, img_width))
            # fflabel = np.fliplr(flabel)
            flabel_classes = np.unique(flabel)

            flag = 0
            for i in range(self.class_num):
                if counts[i] < k and i in flabel_classes:
                    counts[i] += 1
                    flag = 1
                    
            if flag == 1:
                # load img
                fimg_path = fid + '.jpg'
                fpath = os.path.join(self.img_p, fimg_path)
                fimg = resize(imread(fpath) / 255, (img_height, img_width))
                # ffimg = np.fliplr(fimg)
                images.append(fimg)
                # images.append(ffimg)
                labels.append(self.separate_labels(flabel, self.class_num))
                # labels.append(self.separate_labels(fflabel, self.class_num))
            # labels.append(flabel)
                # flag = 1

        images = np.array(images)
        labels = np.array(labels)
        # print(images.shape)
        return images, labels

class CityScapeLoader(DataLoader):
    def __init__(self, img_path, label_path, test_num = 200):
        self.cityscape_labels = {
            'unlabeled':[0,1,2,3,4,5,6, 9, 10,14,15,16,18,29,30],
            'road':[7],
            'sidewalk':[8],
            'building':[11],
            'wall':[12],
            'fence':[13],
            'pole':[17],
            'traffic light':[19],
            'traffic sign':[20],
            'vegetation':[21],
            'terrain':[22],
            'sky':[23],
            'person':[24],
            'rider':[25],
            'car':[26],
            'truck':[27],
            'bus':[28],
            'train':[31],
            'motorcycle':[32],
            'bicycle':[33]
        }
        super().__init__(img_path, label_path)
        self.test_num = test_num
        self.class_num = 20

        self.test_data = self.generate_testing_samples(k=3)

        # a data-loading class for SiftFlow data
        
    def _load(self):
        for root, dirs, files in os.walk(self.label_p):
            for f in sorted(files):
                if f.endswith('.png'):
                    fid = f.split('.')[0]
                    ftype = fid.split('_')[-1]
                    if ftype == 'labelIds':
                        self.img_ids.append(f)
        self.ids = np.arange(len(self.img_ids))
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
        lb[(lb>=0) & (lb<=6)] = 0
        lb[lb==9] = 0
        lb[lb==10] = 0
        lb[(lb>=14) & (lb<=16)] = 0
        lb[lb==18] = 0
        lb[(lb>=29) & (lb<=30)] = 0
        lb[lb==7]=1
        lb[lb==8]=2
        lb[lb==11]=3
        lb[lb==12]=4
        lb[lb==13]=5
        lb[lb==17]=6
        lb[lb==19]=7
        lb[lb==20]=8
        lb[lb==21]=9
        lb[lb==22]=10
        lb[lb==23]=11
        lb[lb==24]=12
        lb[lb==25]=13
        lb[lb==26]=14
        lb[lb==27]=15
        lb[lb==28]=16
        lb[lb==31]=17
        lb[lb==32]=18
        lb[lb==33]=19
        seg_lb = np.zeros((lb.shape[0], lb.shape[1], nc))
        for i in range(nc):
            seg_lb[:,:,i] = (lb==i).astype(int)
        return seg_lb

    def generate_supervised_samples(self,  
                                  k = 1,   # k - number of shots for each class
                                  img_height = 256,
                                  img_width = 256):
        while True:   
            # randomly choose a batch of k images
            yield (self._generate_train(k, img_height, img_width))

    def generate_testing_samples(self, k = 1, 
                                 img_height = 256,
                                 img_width = 256):
        # ids = self.ids[-self.test_num:]
        # load images and corresponding labels
        images = []
        labels = []

        counts = np.zeros(self.class_num)
        for id in np.random.choice(self.ids, size=len(self.ids), replace=False):
            if self.check_datatset_completence(counts, k):
                break
            # print('Loading image of {0}'.format(fid))
            
            # load label
            flabel_path = self.img_ids[id]
            flpath = os.path.join(self.label_p, flabel_path)
            flabel = resize(imread(flpath)[:, :, 0], (256, 256)).astype(int)
            flabel_classes = np.unique(flabel)
            flabel = self.separate_labels(flabel, self.class_num)

            flag = 0
            for i in range(self.class_num):
                if counts[i] < k and i in flabel_classes:
                    counts[i] += 1
                    flag = 1
            if flag==1:
                # load img
                fimg_path = '{0}.jpg'.format(id+1)
                fpath = os.path.join(self.img_p, fimg_path)
                fimg = (imread(fpath) / 255)[:, :256, :]
                images.append(fimg)
                labels.append(flabel)
                # labels.append(flabel)

        images = np.array(images)
        labels = np.array(labels)
        # print(images.shape)
        return images, labels

    def generate_real_samples(self, dataset, n_samples, seed = None):
        imgs, labels = dataset
        if seed:
            np.random.seed(seed)
        # choose random instances
        idx = randint(0, imgs.shape[0], n_samples)
        # load images and labels
        x, y = imgs[idx], labels[idx]
        # generate class labels
        y2 = np.ones([n_samples, x.shape[1], x.shape[2]])
        return [x, y], y2

    # generate labeled training dataset
    def _generate_train(self, k, img_height, img_width): 
        # load images and corresponding labels
        # ids = self.ids[:-self.test_num]
        images = []
        labels = []

        counts = np.zeros(self.class_num)

        for id in np.random.choice(self.ids, size=len(self.ids), replace=False):
            if self.check_datatset_completence(counts, k):
                break
            # print('Loading image of {0}'.format(fid))
            
            # load label
            flabel_path = self.img_ids[id]
            flpath = os.path.join(self.label_p, flabel_path)
            flabel = resize(imread(flpath)[:, :, 0], (256, 256)).astype(int)
            flabel_classes = np.unique(flabel)
            flabel = self.separate_labels(flabel, self.class_num)
            flag = 0
            for i in range(self.class_num):
                if counts[i] < k and i in flabel_classes:
                    counts[i] += 1
                    flag = 1
            if flag == 1:
                # load img
                fimg_path = '{0}.jpg'.format(id+1)
                fpath = os.path.join(self.img_p, fimg_path)
                fimg = (imread(fpath) / 255)[:, :256, :]
                images.append(fimg)
                labels.append(flabel)
                # labels.append(flabel)

        images = np.array(images)
        labels = np.array(labels)
        # print(images.shape)
        return images, labels

class ADKLoader(DataLoader):
    def __init__(self, img_path, label_path, test_num = 100):
        super().__init__(img_path, label_path)
        self.test_num = test_num
        self.class_num = 151
        self.test_data = self.generate_testing_samples()

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
        lb = lb[:, :, 0]
        seg_lb = np.zeros((lb.shape[0], lb.shape[1], nc))
        for i in range(nc):
            seg_lb[:,:,i] = (lb==i).astype(int)
        return seg_lb

    def generate_supervised_samples(self,  
                                  k = 1,   # k - number of shots for each class
                                  img_height = 256,
                                  img_width = 256,
                                  flip = False):
        while True:
            # randomly choose a batch of k images
            yield (self._generate_train(k, img_height, img_width, flip))

    def generate_testing_samples(self, k = 1, 
                                 img_height = 256,
                                 img_width = 256):
        # img_ids = self.img_ids[-self.test_num:]
        # load images and corresponding labels
        images = []
        labels = []

        counts = np.zeros(self.class_num)

        for fid in random.sample(self.img_ids, len(self.img_ids)):
            if self.check_datatset_completence(counts, k):
                break
            # print('Loading image of {0}'.format(fid))
            
            # load label
            flabel_path = fid + '.png'
            flpath = os.path.join(self.label_p, flabel_path)
            flabel = resize(imread(flpath), (img_height, img_width))
            flabel_classes = np.unique(flabel)

            flag = 0
            for i in range(self.class_num):
                if counts[i] < k and i in flabel_classes:
                    counts[i] += 1
                    flag = 1
                    
            if flag==1:
                # load img
                fimg_path = fid + '.jpg'
                fpath = os.path.join(self.img_p, fimg_path)
                fimg = resize(imread(fpath) / 255, (img_height, img_width))
                images.append(fimg)
                labels.append(self.separate_labels(flabel, self.class_num))
                # labels.append(flabel)

        images = np.array(images)
        labels = np.array(labels)
        # print(images.shape)
        return images, labels

    def generate_real_samples(self, dataset, n_samples, seed = None):
        imgs, labels = dataset
        if seed:
            np.random.seed(seed)
        # choose random instances
        idx = randint(0, imgs.shape[0], n_samples)
        # load images and labels
        x, y = imgs[idx], labels[idx]
        # generate class labels
        y2 = np.ones([n_samples, x.shape[1], x.shape[2]])
        return [x, y], y2

    # generate labeled training dataset
    def _generate_train(self, k, img_height, img_width, flip): 
        # load images and corresponding labels
        # img_ids = self.img_ids[:-self.test_num]
        images = []
        labels = []

        counts = np.zeros(self.class_num)

        for fid in random.sample(self.img_ids, len(self.img_ids)):
            if self.check_datatset_completence(counts, k):
                break
            # print('Loading image of {0}'.format(fid))
            
            # load label
            flabel_path = fid + '.png'
            flpath = os.path.join(self.label_p, flabel_path)
            flabel = resize(imread(flpath), (img_height, img_width))
            if flip:
                fflabel = np.fliplr(flabel)
            flabel_classes = np.unique(flabel)

            flag = 0
            for i in range(self.class_num):
                if counts[i] < k and i in flabel_classes:
                    if flip:
                        counts[i] += 2
                    else:
                        counts[i] += 1
                    flag = 1

            if flag == 1:
            # load img
                fimg_path = fid + '.jpg'
                fpath = os.path.join(self.img_p, fimg_path)
                fimg = resize(imread(fpath) / 255, (img_height, img_width))
                if flip:
                    ffimg = np.fliplr(fimg)
                    images.append(ffimg)
                    labels.append(self.separate_labels(fflabel, self.class_num))
                images.append(fimg)
                labels.append(self.separate_labels(flabel, self.class_num))
                # labels.append(flabel)
                    # flag = 1

        images = np.array(images)
        labels = np.array(labels)
        # print(images.shape)
        return images, labels