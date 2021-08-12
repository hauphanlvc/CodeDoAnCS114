import pickle
from flask import Flask, render_template, request
import os
import cv2
import  sys
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np  
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
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
# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "FolderImage"

filename = 'model.sav'
model = pickle.load(open(filename, 'rb'))



# Hàm xử lý reques
@app.route("/", methods=['GET', 'POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
         try:
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Lưu file
                print(image.filename)
                print(app.config['UPLOAD_FOLDER'])
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                print("Save = ", path_to_save)
                image.save(path_to_save)

                # Convert image to dest size tensor
                res = DuDoan1AnhVGG16(model,path_to_save)



                return render_template("index.html", user_image = image.filename , result = res)
                # else:
                #     return render_template('index.html', msg='Không nhận diện được khuôn mặt')
            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('index.html', msg='Hãy chọn file để tải lên')

         except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html', msg='Không nhận diện được khuôn mặt')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)