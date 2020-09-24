import tile_coding
import numpy as np
from matplotlib import pyplot as plt



# construct data
x = np.linspace(-1.0, 1.0, 2000)
y = 10*x**2 * np.sin(3*x)+np.random.normal(0, 0.1, x.shape)

# use tile_coding
tile_num = 20
tile = tile_coding.TileLine(number=tile_num)
# construct model

while True:
    alpha = 0.001
    tile.fit(x, y, alpha)
    y_predict_mean = tile.mean_predict(x)
    y_predict = tile.predict(x)
    plt.scatter(x, y_predict_mean, c='b', s=0.5)
    plt.scatter(x, y_predict, c='k', s=0.5)
    plt.scatter(x, y, c='r', s=0.1)
    plt.draw()
    plt.pause(0.01)
    plt.clf()





