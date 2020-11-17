import tensorflow as tf

class MyModel(tf.keras.Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.flatten = tf.keras.layers.Flatten() #flattens a ND array to a 2D array
            self.input_layer = tf.keras.layers.Dense(10, activation=tf.nn.relu) 
            # self.layer1 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
            self.output_layer = tf.keras.layers.Dense(2) 
        @tf.function
        def call(self, x):
            x = self.flatten(x)
            x = self.input_layer(x)
            # x = self.layer1(x)
            return self.output_layer(x)