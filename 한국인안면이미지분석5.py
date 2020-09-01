import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from activation import mish
import json
from tensorflow.keras.preprocessing import image
from tensorflow import Graph



X=np.load('D:/s/한국인안면/Imageall.npy')



X=X.reshape(360,41,100,100,3)

Y=[]
y = np.arange(0, 41, 1)
for j in y:
    for i in range(360):
        Y.append(j)

y=np.array(Y)
y=y.reshape(360,41)



xtrain,xtest,ytrain,ytest=train_test_split(X,y,train_size=0.75)
xtrain=xtrain.reshape(xtrain.shape[0]*xtrain.shape[1],xtrain.shape[2],xtrain.shape[3],3)
xtest=xtest.reshape(xtest.shape[0]*xtest.shape[1],xtest.shape[2],xtest.shape[3],3)
ytrain=ytrain.reshape(ytrain.shape[0]*ytrain.shape[1],1)
ytest=ytest.reshape(ytest.shape[0]*ytest.shape[1],1)

xtrain=tf.convert_to_tensor(xtrain)
xtest=tf.convert_to_tensor(xtest)
ytrain=tf.convert_to_tensor(ytrain)
ytest=tf.convert_to_tensor(ytest)
#
# # # #--------------------------------------------------------------------------------------------------------------------
#
# from tensorflow.keras.layers import*
# from tensorflow.keras.models import*
#
# model=Sequential()
# model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',input_shape=(xtrain.shape[1],xtrain.shape[2],xtrain.shape[3]),activation='selu',kernel_regularizer=tf.keras.regularizers.l2(l2=0.001)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(filters=256,kernel_size=(3,3),activation='elu',kernel_regularizer=tf.keras.regularizers.l2(l2=0.001)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(filters=512,kernel_size=(3,3),activation='elu',padding='same',kernel_regularizer=tf.keras.regularizers.l2(l2=0.001)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(filters=1024,kernel_size=(3,3),activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l2=0.001)))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(41,activation='softmax'))
#
# from tensorflow.keras.callbacks import ModelCheckpoint
# check = ModelCheckpoint(filepath='D:/s/한국인안면/저장/Epoch_{epoch:03d}_Val_{val_loss:0.3f}.hdf5',monitor='val_loss',save_best_only=True)
# model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'], optimizer='Adam')
# history=model.fit(xtrain,ytrain,epochs=7,validation_split=0.2,callbacks=[check],batch_size=8)
# model.evaluate(xtest,ytest)
#
# model.save('D:/s/한국인안면/modelsave.h5')


# model.predict(x)
#model.save
#25,26을 predict값 c7



# plt.figure(figsize=(12,4))
# plt.subplot(1,2,1)
# plt.plot(history.history['loss'],'b-',label='loss')
# plt.plot(history.history['val_loss'],'r--',label='val_loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.subplot(1,2,2)
# plt.plot(history.history['accuracy'],'g-',label='accuracy')
# plt.plot(history.history['val_accuracy'],'k--',label='val_accuracy')
# plt.ylim(0.7,1)
# plt.legend()
# plt.show()
#



from tensorflow.keras.models import load_model
model=load_model('D:/s/한국인안면/modelsave.h5')
img_height,img_width = 100, 100
testimage = "D:/s/testimage1.jpg"

with open('D:/s/jsonlabeling','r',encoding="utf-8") as f:
    LabelInfo=f.read()
# print(type(LabelInfo))  #str
#
LabelInfo = json.loads(LabelInfo)
#

img=image.load_img(testimage,target_size=(img_height,img_width))
x = image.img_to_array(img)
x = x.reshape(1,img_height,img_width,3)
predi = model.predict(x)
a=np.argmax(predi)
b=LabelInfo['{}'.format(a)]
print(b)
# print(LabelInfo('{}'.format(np.argmax(predi))))






