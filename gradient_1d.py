import tile_coding
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from matplotlib import pyplot as plt

# construct data
x = np.linspace(-1.0, 1.0, 2000)
y = 10*x**2 * np.sin(3*x)+np.random.normal(0, 0.1, x.shape)

# use tile_coding
tile = tile_coding.Tile1D(number=10)

# construct model
lines = []
for i in range(tile.number):
    input_x = Input(shape=[1])
    o = layers.Dense(1)(input_x)
    line = Model(input_x, o)
    line.compile(optimizer=tf.optimizers.SGD(learning_rate=0.1, momentum=0.9), loss=tf.losses.mean_squared_error)
    lines.append(line)


while True:

    tile_index = tile.x(x)[1]
    for tile_index_count in range(tile.number):
        x_sort = x[np.where(tile_index == tile_index_count)]
        y_sort = y[np.where(tile_index == tile_index_count)]
        local_model = lines[tile_index_count]
        local_model.fit(x=x_sort, y=y_sort, batch_size=32, epochs=1)

        plt.scatter(x_sort, local_model(x_sort), c='b')
        plt.scatter(x_sort, y_sort, c='r', s=0.1)
    plt.draw()
    plt.pause(0.001)
    plt.clf()




