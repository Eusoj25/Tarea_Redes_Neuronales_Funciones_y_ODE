import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np 

class REDfunciones(Sequential):
    def __init__(self, **kwars):
        super().__init__(**kwars)
        self.loss_tracker = keras.metrics.Mean(name = "loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        x = tf.random.uniform((batch_size,1), minval= -1, maxval= 1)

        with tf.GradientTape() as tape:
            #Compute the loss value
            y_pred = self(x, training=True)
            eq = y_pred - (1 + 2*x + 4*x**3)
            loss = keras.losses.mean_squared_error(0., eq)
            

        # Apply grads
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #update metrics
        self.loss_tracker.update_state(loss)
        #Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result()}

model = REDfunciones()

model.add(Dense(10, activation='tanh', input_shape = (1,)))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=RMSprop(), metrics=["loss"])

x = tf.linspace(-1,1,100)
history = model.fit(x,epochs = 1500, verbose = 1)

x_testv = tf.linspace(-1,1,100)
a = model.predict(x_testv)
plt.plot(x_testv,a)
plt.plot(x_testv,1 + 2*x + 4*x**3)
plt.suptitle('Aproximación de una función con una RNA')
leyendas = ['y_model(x)','y(x)']
plt.legend(loc = "upper right", labels = leyendas)
plt.show()




            


