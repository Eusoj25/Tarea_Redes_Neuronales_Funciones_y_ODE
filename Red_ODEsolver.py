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
        x_0 = tf.zeros((batch_size,1))

        with tf.GradientTape() as tape:
            #Compute the loss value
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)
                g.watch(x_0)

                with tf.GradientTape() as gg:
                    gg.watch(x)    
                    y_pred = self(x, training=True)
                y_x = gg.gradient(y_pred,x)
                y_0 = self(x_0, training=True)

            y_xx = g.gradient(y_x, x)
            y_xo = g.gradient(y_0,x_0)
            eq = y_xx + y_pred
            ic = y_0 - 1.0 
            ic1 = y_xo + 0.5
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic) + keras.losses.mean_squared_error(0., ic1)

        # Apply grads
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #update metrics
        self.loss_tracker.update_state(loss)
         #Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result()}

model = ODEsolver()

model.add(Dense(10, activation='tanh', input_shape = (1,)))
model.add(Dense(20, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(20, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(20, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(20, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=RMSprop(learning_rate=0.0001), metrics=["loss"])

x = tf.linspace(-5,5,100)
history = model.fit(x,epochs = 3000, verbose = 1)

x_testv = tf.linspace(-5,5,100)
a = model.predict(x_testv)
plt.plot(x_testv,a)
plt.plot(x_testv, np.cos(x) -0.5* tf.math.sin(x) )
plt.suptitle('Solución de una ODE con una RNA')
leyendas = ['y_model(x)','y(x)']
plt.legend(loc = "upper right", labels = leyendas)
plt.show()