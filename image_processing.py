import numpy as np
import pickle
import random
import os
import cv2

# this is where you store your datasets for this project
DataDir = "G:\AI\Dog-or-Cat-Datasets"
Categories = ["Dog","Cat"]

# read the images with labels as arrays for later
training_data = []
image_size = 70
def pre_process_Data():
    for Category in Categories:
        # class_name = Categories.index(Category)
        if Category == "Dog":
            label = 0
        else:
            label = 1
        path = os.path.join(DataDir,Category)
        for img in os.listdir(path):
            try:         
                img_array = cv2.imread(os.path.join(path, img))
                img_array = np.array(img_array)
                img_array = cv2.resize(img_array,(image_size, image_size))
                img_array = img_array/255.0                                #normalization
                training_data.append((img_array, label))
            except Exception as e:
                pass

n=pre_process_Data()

# Save the datasets we processed and divide it into two parts(one for training,one for testing)
random.shuffle(training_data)
x, y = [], []
for img,label in training_data:
    x.append(img)
    y.append(label)
x = np.array(x)
x_train = x[:20000]
y_train = y[:20000]
x_test = x[20000:]
y_test = y[20000:]

#save data for trainning
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


