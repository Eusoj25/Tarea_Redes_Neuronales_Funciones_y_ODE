import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np 

class ODEsolver(Sequential):
    def __init__(self, **kwars):
        super().__init__(**kwars)
        self.loss_tracker = keras.metrics.Mean(name = "loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        x = tf.random.uniform((batch_size,1), minval= -5, maxval= 5)

        with tf.GradientTape() as tape:
            #Compute the loss value
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                y_pred = self(x, training=True)
            dy = tape2.gradient(y_pred,x)
            x_0 = tf.zeros((batch_size,1))
            y_0 = self(x_0, training=True)
            eq = x*dy + y_pred - (x**2)*tf.math.cos(x)
            ic = y_0 - 0
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic)

        # Apply grads
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #update metrics
        self.loss_tracker.update_state(loss)
        #Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result()}

model = ODEsolver()

model.add(Dense(10, activation='tanh', input_shape = (1,)))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=RMSprop(learning_rate=0.0001), metrics=["loss"])

x = tf.linspace(-5,5,100)
history = model.fit(x,epochs = 1500, verbose = 1)

x_testv = tf.linspace(-5,5,100)
a = model.predict(x_testv)
plt.plot(x_testv,a)
plt.plot(x_testv, x*tf.math.sin(x)+2*tf.math.cos(x)-2*tf.math.sin(x)/x)
plt.suptitle('Solución de una ODE con una RNA')
leyendas = ['y_model(x)','y(x)']
plt.legend(loc = "upper right", labels = leyendas)
plt.show()