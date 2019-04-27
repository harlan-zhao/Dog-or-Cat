from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

# name for tensorboard logs
name = "2conv-10epoch"
tensorboard = TensorBoard(log_dir="logs/{}".format(name))

# load and modify(numpy.array) data
x = pickle.load(open("x_train.pickle","rb"))
y = pickle.load(open("y_train.pickle","rb"))
x_test = pickle.load(open("x_test.pickle","rb"))
y_test = pickle.load(open("y_test.pickle","rb"))
x_test = np.array(x_test)
x = np.array(x)

#CNN model(2_conv_layers,using maxpooling with stride of (2,2))
model = Sequential()
model.add(Conv2D(64,(3,3),input_shape = x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"],)

model.fit(x,y,batch_size=32,epochs=10,validation_split=0.2,callbacks=[tensorboard])

#out of sample validation
val_loss, val_acc = model.evaluate(x_test,y_test)

#save model
model.save("CNN_for_catdog")
