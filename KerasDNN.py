import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train=tf.keras.utils.normalize(x_train, 1)
x_test=tf.keras.utils.normalize(x_test, 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

tbCallBack=tf.keras.callbacks.TensorBoard(log_dir="output/KerasDNN", histogram_freq=0, write_graph=True, write_images=True)
model.fit(x_train, y_train, epochs=3, callbacks=[tbCallBack])

model.save('model/KerasDNN.h5')

# model = tf.keras.models.load_model('my_model.h5')

#
# plt.imshow(x_train[0],cmap=plt.cm.binary)
# plt.show()