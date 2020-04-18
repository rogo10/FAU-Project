import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
from tensorflow.keras import models
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Conv1D, Flatten, Activation, MaxPooling1D
import numpy as np
import pandas as pd
import skimage.io as io
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import classification_report, confusion_matrix

#silents warning from tensorflow
#more on it here: https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#get in photos for cnn
def get_photos(path):
    images=io.ImageCollection(path)
    print(len(images),"images found")
    my_images=[]
    for i in range(len(images)):
        img=np.array(images[i])
        img=img/255
        my_images.append(img) 
    
    return np.array(my_images)


path1=r'1_resized/*.tif'
dit=get_photos(path1)
path0=r'0_resized/*.tif'
ndit=get_photos(path0)
print(dit.shape)
label1 = np.array([1]*len(dit)).reshape(-1,1)
label0 = np.array([0]*len(ndit)).reshape(-1,1)
y=np.vstack((label1,label0))
#print(y)
X=np.vstack((dit,ndit))
#X=scale(X)

print("Shape of X:",X.shape)
print("Shape of y:",y.shape)



x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

print("Shape of x_train:",x_train.shape)
print("Shape of y_train:",y_train.shape)


#create model
model = Sequential()
model.add(BatchNormalization())
model.add(layers.Conv1D(filters=625,kernel_size=3,padding='same',activation='relu',input_shape=(500,500)))
model.add(MaxPooling1D(pool_size=10))
#model.add(layers.Conv1D(filters=125,kernel_size=3,padding='same',activation='relu'))
#model.add(MaxPooling1D(pool_size=10))
model.add(Dropout(0.25))
#model.add(Dense(12))
model.add(Flatten())
model.add(Dense(12,activation="relu"))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(2,activation="softmax"))
#print(model.summary())
print(model)


#test model
model.compile(optimizer='adamax',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
                )
model.fit(x_train,y_train,epochs=15,validation_data=(x_test,y_test))
test_loss, test_acc = model.evaluate(x_test,y_test,verbose=2)
print(test_acc)

pred = model.predict(x_test)
pred = np.argmax(np.round(pred),axis=1)
target_names = ["Class {}".format(i) for i in range(2)]
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred,target_names=target_names))
















