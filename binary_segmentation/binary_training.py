import numpy as np
import tensorflow as tf
from keras.utils import Sequence
from albumentations import ShiftScaleRotate, Flip, GridDistortion
from skimage import morphology
import os
from glob import glob
import cv2
from binary_model import model
import argparse

parser = argparse.ArgumentParser(
                    prog='Binary Segmentation Inference',
                    description='This program runs a keras model on x-ray coronary angiogram to extract vessel structures')

parser.add_argument('--path', '-p', default=r'imgs\train')
args = parser.parse_args()


class MyGenerator(Sequence):
    def __init__(self, imgs, msks, weights, batch_size = 2, to_fit = True, train = True, shuffle = True):
        self.batch_size = batch_size
        self.idxs = np.arange(imgs.shape[0])
        self.to_fit = to_fit
        self.train = train
        self.shuffle = shuffle
        self.imgs = imgs 
        self.msks = msks
        self.class_weights = weights  
        
    def __len__(self):
        return int(np.ceil(self.idxs.shape[0] / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)
    
    def load_Xy(self,batch):
        X = []
        y = []
        w = [] 
        for i in batch:
            img = tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(args.path+f'\images\{self.imgs[i]}', 
                                                    color_mode='grayscale', 
                                                    target_size=(512,512))
                                                     )
            
            
            msk = tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(args.path+f'\masks\{self.msks[i]}', 
                                                    color_mode='grayscale', 
                                                    target_size=(512,512))
                )
            img = np.squeeze(img)
            clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8)) 
            img_not = cv2.bitwise_not(img) 
            se = np.ones((50,50), np.uint8) 
            wth = morphology.white_tophat(img_not, se) 
            raw_minus_topwhite = img.astype(int) - wth
            raw_minus_topwhite = ((raw_minus_topwhite>0)*raw_minus_topwhite).astype(np.uint8) 
            img = clahe.apply(raw_minus_topwhite)
            img = img[:,:,np.newaxis]
            if self.train:
                aug = Flip(p=0.5)
                transform = aug(image = img, mask = msk)
                aug = ShiftScaleRotate(always_apply=True, border_mode = 2)
                transform = aug(image = transform['image'], mask = transform['mask'])
                aug = GridDistortion(p = 0.5,)
                transform = aug(image = transform['image'], mask = transform['mask'])
                img = transform['image']
                msk = transform['mask']
            img = img/255
            msk = (msk>0)*1
            we = np.abs(np.abs(img-1)-msk)

            y.append(msk) 
            X.append(img)
            w.append(we)

            

        X = np.array(X)
        w = np.array(w)
        y = np.array(y)

        return X, y, w
    
    def __getitem__(self, idx):
        if idx == self.__len__()-1:
            batch = self.idxs[idx * self.batch_size:]
        else:
            batch = self.idxs[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.to_fit:
            return self.load_Xy(batch)
        
model = model()
model.compile(loss={'seg' : tf.keras.losses.SparseCategoricalCrossentropy()}, optimizer = 'Adam', sample_weight_mode="temporal") 

imgs = [os.path.basename(i) for i in glob(args.path+f'\images\*')]
imgs = np.array(imgs)
msks = [os.path.basename(i) for i in glob(args.path+f'\masks\*')]
msks = np.array(msks)

train_gen = MyGenerator(imgs[:int(imgs.shape[0]*0.5)], msks[:int(imgs.shape[0]*0.5)], weights = [1,80], batch_size = 2)
valid_gen = MyGenerator(imgs[int(imgs.shape[0]*0.5):int(imgs.shape[0]*0.9)], msks[int(imgs.shape[0]*0.5):int(imgs.shape[0]*0.9)], weights = [1,80], train = False)
test_gen = MyGenerator(imgs[int(imgs.shape[0]*0.9):], msks[int(imgs.shape[0]*0.9):], weights = [1,80], train = False, shuffle = False)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights = True)

model.fit(train_gen, validation_data = valid_gen, epochs = 1000, callbacks = [callback])