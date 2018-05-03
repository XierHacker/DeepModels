from vgg_net import VGG16
import tensorflow as tf
import numpy as np

train=np.random.normal(size=(100,224,224,3))
print(train.shape)

labels=np.random.random(size=100)*4
labels=labels.astype(np.int)


model=VGG16()
model.fit(train,labels,epochs=10,batch_size=10)