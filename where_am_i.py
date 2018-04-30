
# coding: utf-8

# In[5]:


import numpy as np  
import pandas as pd  
import scipy.stats   
import scipy.special  
import subprocess
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline  
import sklearn.preprocessing  
import sklearn.ensemble  
import sklearn.kernel_ridge 
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
# import pandas_profiling
import os
#simpsons=glob.glob('C:\backup\simpsons_dataset.tar.gz\simpsons_dataset\abraham_grampa_simpson\*.jpg')
from os import listdir
from os.path import isfile, join


# In[6]:



import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
#from keras import model
import keras
from keras.models import load_model
#from keras.applications.inception_v3 import InceptionV3

from keras.applications.inception_v3 import InceptionV3
#keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.applications.inception_v3 import get_source_inputs

print (get_source_inputs)



# model=InceptionV3()
# print (model.summary())


# In[7]:


# listdir(mypath)


# In[8]:


import glob
glob.glob('train/opencountry/'+'*.jpg')


# In[9]:



mypath='train/'
image_all=[]
onlyfiles=[]
imgfiles=[] 
imgfiles_dict={}
print (type(mypath))
for f in listdir(mypath):
    imgfiles=[]
    if isfile:            
        image_all.append(glob.glob('train/'+f+'/*.jpg'))
        imgfiles.append(glob.glob('train/'+f+'/*.jpg'))
        imgfiles_dict.update({f:imgfiles})


# In[10]:


image_all=np.array(image_all)


# In[11]:


image_all[1][0]


# In[12]:


# FD.shape


# In[13]:


iii=cv2.imread(image_all[1][2],0)
plt.imshow(iii)


# In[14]:


img_sample=cv2.imread(image_all[0][1])
img_sample.shape


# # 列出file name , 照片長度
# 

# In[15]:


file_list=[s for key,s in enumerate(imgfiles_dict)]


# In[16]:


list_a=[6,9,4,1,14,3,5,13,0,11,8,2,12,10,7]


# In[17]:


Ls_file=pd.DataFrame(np.array(file_list),np.array(list_a))


# In[18]:


len_pic=[np.array(imgfiles_dict[file_list[i]]).reshape(-1).shape for i in range(len(file_list)) ]
len_pic


# In[19]:


np.array(imgfiles_dict[file_list[0]])[0][0]


# In[20]:


for i in range(len(image_all)):
    img=cv2.imread(np.array(imgfiles_dict[file_list[i]])[0][0])
    img=cv2.resize(img,(224,224))
    if img.shape!=(224,224,3):
        print(img.shape)


# In[21]:


image_all[13][233]


# In[22]:


img=cv2.imread(image_all[0][309])
plt.imshow(img)


# In[23]:


Ls_file


# In[24]:


image_all.shape[0]


# In[25]:


immg=[]#np.array([]).reshape((0,img_sample.shape[0],img_sample.shape[1],img_sample.shape[2]))
for j in range(image_all.shape[0]):
    for i in range(len(image_all[j])):
        img=cv2.imread(image_all[j][i])
        img=cv2.resize(img,(224,224))
        img=img.reshape(-1,img.shape[0],img.shape[1],img.shape[2])
        immg.append(img)


# In[26]:


len(immg)


# In[27]:


im=np.array(immg).reshape(2985,224,224,-1)


# In[28]:


im.shape

im.shape[0]
# In[29]:


for i in range(len(len_pic)):
    print (len_pic[i][0])


# In[30]:


label=np.zeros((im.shape[0],1))
sum1=0
for i in range(len(len_pic)):
    if i ==0:
        print(i)
        label[:len_pic[i][0]]=Ls_file.index[i]
        sum1+=len_pic[i][0]
    else :
        print(sum1,(sum1+len_pic[i][0]))
        label[sum1:(sum1+len_pic[i][0])]=Ls_file.index[i]
        sum1+=len_pic[i][0]


# In[31]:


label[2870: 2985]


# In[32]:


print('OK')


# In[33]:


Y=pd.get_dummies(label.reshape(-1))


# In[34]:


im_normal=im/255.


# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(im_normal, Y, test_size=0.2, random_state=42)


# In[36]:


X_train[0].shape#


# # Model

# In[37]:


from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ReduceLROnPlateau
augRatio = 5  #Data augmentation時，要產生幾倍數量的dataset
random_seed = 8
batch_size = 20  #每批次的數量
epochs = 30  #訓練時要跑的世代次數
from keras.models import load_model


# In[38]:


epochs


# In[39]:


dataAugment=True
if(dataAugment):

    train_datagen = ImageDataGenerator(

        zoom_range = 0.2,

        height_shift_range = 0.2,

        width_shift_range = 0.2,

        shear_range=0.1,
        

        rotation_range = 40,
        horizontal_flip=False)

    train_datagen.fit(X_train)
    


# In[40]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=4, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[37]:



model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', input_shape=(256, 256,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.5))


model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))




model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.5))



model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=15, kernel_initializer='normal', activation='softmax'))
print(model.summary())


# In[38]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
steps_per_epoch=int((len(X_train)*augRatio)/batch_size)


# In[39]:


# model=load_model("my_model20.h5")


# In[40]:


batch_size=60
for i in range(20,31):

    train_history = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),

                        steps_per_epoch=len(X_train)/40, epochs=10,

                        validation_data=(X_test, y_test), verbose = 1)#,callbacks=[learning_rate_reduction])

    model.save("my_model"+str(i)+".h5")
# else:
#     train_history = model.fit(x=x_Train_normalize, y=y_TrainOneHot, validation_split=0.2, epochs=20, batch_size=200, verbose=2)


# # Model Predict

# In[58]:


mo=load_model("my_model_In0.h5")


# In[ ]:


mo.summary()


# In[ ]:


#  train_history = mo.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),

#                         steps_per_epoch=len(X_train)/10, epochs=10,

#                         validation_data=(X_test, y_test), verbose = 1)#,callbacks=[learning_rate_reduction])


# In[50]:


A=glob.glob('testset/*.jpg')[0]
print (glob.glob('testset/*.jpg')[0].split(".jpg")[0].split("/")[1])


# In[51]:


img_t=cv2.imread(glob.glob('testset/*.jpg')[1],0)
plt.imshow(img_t)


# In[52]:


img_t.shape


# In[53]:


test_img=[]
for i in range(len(glob.glob('testset/*.jpg'))):
    img_t=cv2.imread(glob.glob('testset/*.jpg')[i])
    img_t=cv2.resize(img_t,(224,224))
    test_img.append(img_t)


# In[54]:


test_name=[glob.glob('testset/*.jpg')[i].split(".jpg")[0].split("/")[1] for i in range(len(glob.glob('testset/*.jpg')))]


# In[55]:


test_img=np.array(test_img)


# In[56]:


test_img_n=test_img/255


# In[57]:


test_img_n=test_img_n.reshape(test_img_n.shape[0],test_img_n.shape[1],test_img_n.shape[2],-1)


# In[58]:


test_img_n.shape


# In[79]:


Y_test_img=model.predict(test_img_n)


# In[80]:


results = np.argmax(Y_test_img,axis = 1)


# In[81]:


Ls_file


# In[82]:


test_final=pd.DataFrame([np.array(test_name),results]).T


# In[83]:


test_final=test_final.rename(columns={0:'id',1:'class'})


# In[84]:


sub.columns


# In[85]:


sub=pd.read_csv('submission_cnn.csv')[['id']]


# In[86]:


# test_final


# In[67]:


check=pd.merge(sub,test_final,on=['id'])


# In[87]:


check.to_csv('test_cnn7.csv',index=False)


# # transfer learning

# In[45]:


print('start')


# In[42]:


from keras.layers import AveragePooling2D,GlobalAveragePooling2D


# In[ ]:


from keras.applications.resnet50 import ResNet50 
from keras.applications.xception import Xception
from keras.preprocessing import image 
from keras.applications.resnet50 import preprocess_input, decode_predictions 
from keras.optimizers import Adam 
from keras.layers.normalization import BatchNormalization

from keras import Sequential

base_model = Xception(weights='imagenet', include_top =False, input_shape = (224,224, 3)) 
model = Sequential() 
model.add(base_model) 
model.add(Flatten()) 
model.add(Dense(512, activation="relu")) 
model.add(Dropout(0.2))
model.add(Dense(256, activation="relu")) 

model.add(Dense(units=15, activation='softmax')) 
for layer in base_model.layers[:-1]: 
    layer.trainable = False 
    
base_model.layers[-1].trainable

optimizer = Adam(lr=1e-4) 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.summary()


# In[ ]:



# optimizer = Adam(lr=1e-4) 
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 


# In[ ]:


batch_size = 60  #每批次的數量

for i in range(0,20):

    train_history = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),

                        steps_per_epoch=len(X_train)/50, epochs=10,

                        validation_data=(X_test, y_test), verbose = 1)#,callbacks=[learning_rate_reduction])

#     model.save("my_model_inception"+str(i)+".h5")


# In[ ]:


# custom_resnet_model.layers[-1].trainable


# # Model1

# In[47]:


from keras.applications.inception_v3 import InceptionV3


# In[48]:


image_input = Input(shape=(256, 256, 3))

model_In = InceptionV3(input_tensor=image_input, include_top=False,weights='imagenet')

model_In.summary()


# In[ ]:


print('l')


# In[55]:


last_layer = model_In.get_layer('mixed10').output
x   = Flatten(name='flatten')(last_layer)
out = Dense(15, activation='softmax', name='output_layer')(x)
custom_resnet_model = Model(inputs=image_input,outputs= out)
custom_resnet_model.summary()

for layer in custom_resnet_model.layers[:-1]:
    layer.trainable = False

custom_resnet_model.layers[-1].trainable

custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[56]:


batch_size = 60  #每批次的數量

for i in range(0,20):

    train_history = custom_resnet_model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),

                        steps_per_epoch=len(X_train)/50, epochs=10,

                        validation_data=(X_test, y_test), verbose = 1)#,callbacks=[learning_rate_reduction])

    custom_resnet_model.save("my_model_In"+str(i)+".h5")


# In[ ]:



hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


# In[ ]:


num_classes=15
for layer in custom_resnet_model.layers[:-1]:
    layer.trainable = False

custom_resnet_model.layers[-1].trainable

custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

t=time.time()
hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


# In[57]:


# num_classes=15
# last_layer = model.output
# # add a global spatial average pooling layer
# x = GlobalAveragePooling2D()(last_layer)
# x = Dropout(0.5)(x)
# # add fully-connected & dropout layers
# x = Dense(512, activation='relu',name='fc-1')(x)
# x = Dropout(0.5)(x)
# x = Dense(256, activation='relu',name='fc-2')(x)
# x = Dropout(0.5)(x)
# # a softmax layer for 4 classes
# out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# # this is the model we will train
# custom_resnet_model2 = Model(inputs=model.input, outputs=out)

# custom_resnet_model2.summary()# 


# In[58]:


for layer in custom_resnet_model2.layers[:-6]:
    layer.trainable = False

custom_resnet_model2.layers[-1].trainable


# In[59]:


print('ok')# custom_resnet_model2=load_model('my_model_inception6.h5')


# In[60]:


from keras import optimizers
# adam1=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
custom_resnet_model2.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[61]:


batch_size = 60  #每批次的數量

for i in range(0,20):

    train_history = custom_resnet_model2.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),

                        steps_per_epoch=len(X_train)/50, epochs=10,

                        validation_data=(X_test, y_test), verbose = 1)#,callbacks=[learning_rate_reduction])

    custom_resnet_model2.save("my_model_inception"+str(i)+".h5")


# # model2

# In[50]:


# Fine tune the resnet 50
image_input = Input(shape=(198, 198, 3))
model = ResNet50(input_shape=(198,198, 3),weights='imagenet',include_top=False)


# In[51]:


num_classes=15
last_layer = model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
# x = Dense(512, activation='relu',name='fc-1')(x)
# x = Dropout(0.5)(x)
x = Dense(256, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
# a softmax layer for 4 classes
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
custom_resnet_model2 = Model(inputs=model.input, outputs=out)

custom_resnet_model2.summary()


# In[52]:


for layer in custom_resnet_model2.layers[:-6]:
    layer.trainable = False

custom_resnet_model2.layers[-1].trainable

custom_resnet_model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[53]:


batch_size = 60  #每批次的數量

for i in range(0,20):

    train_history = custom_resnet_model2.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),

                        steps_per_epoch=len(X_train)/15, epochs=10,

                        validation_data=(X_test, y_test), verbose = 1)#,callbacks=[learning_rate_reduction])

    custom_resnet_model2.save("my_model_resnet"+str(i)+".h5")


# In[ ]:


X_train, X_test, y_train, y_test


# In[ ]:


epochs


# In[57]:


custom_model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[59]:


hist = custom_model.fit(X_train, y_train, batch_size=52, epochs=12, verbose=1, validation_data=(X_test, y_test))
(loss, accuracy) = custom_model.evaluate(X_test, y_test, batch_size=50, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

