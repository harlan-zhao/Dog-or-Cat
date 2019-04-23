import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DataDir = "G:\AI\Dog-or-Cat-Datasets"    # this is where you store your datasets for this project
Categories=["Dog","Cat"]
for Category in Categories:
    path=os.path.join(DataDir,Category)  # define path for each of the category
    for img in os.listdir(path):         # read the images from path as arrays
        img_array=cv2.imread(img) 

training_data=[]
image_size=50

def pre_process_Data():
    for Category in Categories:
        class_name=Categories.index(Category)
        path=os.path.join(DataDir,Category) 
        for img in os.listdir(path):
            try:         
                img_array=cv2.imread(os.path.join(path,img),0)
                img_array=np.array(img_array)
                # img_array =img_array.reshape(-1,image_size,image_size,1)
                img_array=img_array/255.0
                img_array=cv2.resize(img_array,(image_size,image_size))
                training_data.append([img_array,class_name])

            except:
                pass
pre_process_Data()

# Save the datasets we processed and divide it into two parts(one for training,one for testing)
import pickle
import random
random.shuffle(training_data)
x,y=[],[]
for img,label in training_data:
    x.append(img)
    y.append(label)
x=np.array(x).reshape(-1,image_size,image_size,1)
x_train=x[:20000]
y_train=y[:20000]
x_test=x[20000:]
y_test=y[20000:]
pickle_out = open("x_train.pickle","wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open("x_test.pickle","wb")
pickle.dump(x_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()


