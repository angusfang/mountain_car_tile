import numpy as np
import tile_coding
from matplotlib import pyplot as plt

# create training data
x1 = np.linspace(-1, 1, 50)
x2 = np.linspace(-1, 1, 50)
x1, x2 = np.meshgrid(x1, x2)
y = np.cos(x1) + np.cos(x2)
x1 = np.reshape(x1, [-1])
x2 = np.reshape(x2, [-1])
y = np.reshape(y, [-1])

# use tile function
tile = tile_coding.TilePlane2(domain=[[-1.0, 1.0], [-1.0, 1.0]], number=[5, 5])

while True:
    alpha = 0.001
    tile.fit(x1, x2, y, alpha)
    y_predict_mean = tile.mean_predict(x1, x2)
    y_predict = tile.predict(x1, x2)
    if 'fig' not in locals().keys():
        fig = plt.figure()
    axis = fig.gca(projection='3d')
    axis.scatter3D(x1, x2, y_predict, c=y_predict, cmap='Greys')
    axis.scatter3D(x1, x2, y_predict_mean, c=y_predict_mean, cmap='Purples', s=5)
    plt.pause(0.1)
    fig.clf()


