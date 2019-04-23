import tensorflow as tf
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tkinter import messagebox
import tkinter
import random
x_test=pickle.load(open("x_test.pickle","rb"))
y_test=pickle.load(open("y_test.pickle","rb"))
img_num=random.randint(0,4000)
model=tf.keras.models.load_model("CNN_for_catdog")
categories=["dog","cat"]

img_num = random.randint(0, 4000)
predictions = model.predict(x_test[img_num].reshape(1, 50, 50, 1))
print(predictions[0][0], y_test[img_num])
temp = np.squeeze(x_test[img_num], axis=2)
plt.imshow(temp, cmap="gray")
plt.show()


root=tkinter.Tk()
root.withdraw()
res=int(round(predictions[0][0]))
if res>=0.5:
    res=1
else:
    res=0
messagebox.showinfo("Prediction","The animal on the picture is a {}".format(categories[res]))

