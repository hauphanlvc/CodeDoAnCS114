import pickle

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

model = VGG16(weights='imagenet', include_top=False)
def TrichXuatFeatures(file):
  img = image.load_img(file, target_size=(224, 224)) # chuyển ảnh về size (224,224)
  x = image.img_to_array(img)        # chuyển ảnh về thành 1 array
  x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
  x = preprocess_input(x)
  features = model.predict(x)
  features = np.array(features).reshape(-1,1)
  return features
def DuDoan1AnhVGG16(model,file): # dự đoán 1 ảnh dựa theo cách trích xuất VGG16
  X = []
  features = TrichXuatFeatures(file)
  X.append(features)
  X = np.array(X)
  dimX1_, dimX2_, dimX3_ =X.shape
  X = np.reshape(np.array(X), (dimX1_, dimX2_*dimX3_))
  y_pred = model.predict(X)
  return y_pred
filename = 'model.sav'
model_ = pickle.load(open(filename, 'rb'))
print(DuDoan1AnhVGG16(model_,"FolderImage/DinhDocLap.jpg"))