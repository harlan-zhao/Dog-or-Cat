import tensorflow as tf
import pickle
from matplotlib import pyplot as plt
from tkinter import messagebox
import tkinter
import random

#load data
img_size=70
x_test=pickle.load(open("x_test.pickle","rb"))
y_test=pickle.load(open("y_test.pickle","rb"))

#generate a random index of test img
img_num=random.randint(0,4000)

#load model
model=tf.keras.models.load_model("CNN_for_catdog")

categories=["dog","cat"]

#reshape data to fit model
img_pre=x_test[img_num].reshape(-1, img_size, img_size, 3)

#get predictions
predictions = model.predict([img_pre])  # passing in a list is required

#plot iamge used for prediction
plt.imshow(x_test[img_num])
plt.show()

#messagebox for showing the predict result
root=tkinter.Tk()
root.withdraw()
res=int(round(predictions[0][0]))
messagebox.showinfo("Prediction","The animal on the picture is a {}".format(categories[res]))

