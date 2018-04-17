import cv2
from keras.models import Model
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,Activation, Dropout, Flatten, Dense,UpSampling2D,Input,Concatenate,add
import keras
from keras.models import Model, Sequential, load_model
import numpy as np


def visualize(img):
     img = np.squeeze(img,axis=0)
     #import pdb;pdb.set_trace()

     max_img = np.max(img)
     min_img = np.min(img)
     img = img-(min_img)
     img=img/(max_img - min_img)
     img = img*255 
    # img = img.reshape(img.shape[:2])
     cv2.imwrite('layer1_noreshape_filter'+str(filter1)+'_'+str(ke_width)+'x'+str(ke_height)+'.jpg',img)
     
girl = cv2.imread('girl.jpg')
model = Sequential()
ke_width = 3
ke_height= 3
filter1 = 6
model.add(Conv2D(filter1,ke_width,ke_height,kernel_initializer = keras.initializers.Constant(value=0.12),input_shape= girl.shape,name='conv_1'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(filter1,ke_width,ke_height,input_shape= girl.shape,name='conv_2'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(filter1,ke_width,ke_height,input_shape= girl.shape,name='conv_3'))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(filter1,ke_width,ke_height,input_shape= girl.shape,name='conv_4'))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(8, activation='relu',name='dens_1'))
model.save_weights('girl.h5')
# only load the first layer's weights
model2 = Sequential()
model2.add(Conv2D(filter1,ke_width,ke_height,input_shape= girl.shape,name='conv_1'))
model2.add(MaxPooling2D(pool_size=(3,3)))
model2.add(Activation('relu'))
model2.load_weights('girl.h5', by_name=True)



girl_batch = np.expand_dims(girl,axis=0)
conv_girl = model2.predict(girl_batch)
visualize(conv_girl)

