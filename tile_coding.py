from collections import Iterable

import numpy as np


class Tile1D:
    def __init__(self, domain=[-1.0, 1.0], number=50):
        self.domain = domain
        self.number = number

    def x(self, s):
        interval = (self.domain[1] - self.domain[0]) / self.number
        index = (s - self.domain[0]) // interval
        index = index.astype(int)
        for i, v in enumerate(index):
            assert v <= self.number, 'index must smaller than self.number'
            if v == self.number:
                index[i] -= 1
        tile = np.zeros([self.number * 2, s.shape[0], 1])

        for i in range(s.shape[0]):
            tile[index[i], i, 0] = s[i]  # for weight
            tile[index[i] + self.number, i, 0] = 1  # for bia

        return tile, index


class Tile2D:
    def __init__(self, tile1, tile2):
        self.tile1 = tile1
        self.tile2 = tile2

    def x(self, s1, s2):

        interval1 = (self.tile1.domain[1] - self.tile1.domain[0]) / self.tile1.number
        index1 = (s1 - self.tile1.domain[0]) // interval1
        index1 = index1.astype(int)
        for i, v in enumerate(index1):
            assert v <= self.tile1.number, 'index must smaller than self.number'
            if v == self.tile1.number:
                index1[i] -= 1

        interval2 = (self.tile2.domain[1] - self.tile2.domain[0]) / self.tile2.number
        index2 = (s2 - self.tile2.domain[0]) // interval2
        index2 = index2.astype(int)
        for i, v in enumerate(index2):
            assert v <= self.tile2.number, 'index must smaller than self.number'
            if v == self.tile2.number:
                index2[i] -= 1

        tile = np.zeros([s1.shape[0], self.tile1.number, self.tile2.number])
        for i in range(s1.shape[0]):
            tile[i, index1[i], index2[i]] = s1[i] + s2[i]
        return tile


# use weights and can use mean predict
class TileLine:
    def __init__(self, domain=[-1.0, 1.0], number=10):
        self.domain = domain
        self.number = number
        self.weights = np.zeros([number, 2])
        self.x_mean = np.zeros([number, 1])

        interval = (self.domain[1] - self.domain[0]) / self.number
        count = 0
        for i in range(number):
            self.x_mean[i] = count * interval + 0.5 * interval + self.domain[0]
            count += 1

    # can use raw x_array to predict y
    def predict(self, x_input):

        tis = self.find_index(x_input)

        weights = self.weights
        y_predict = weights[tis, 0] + weights[tis, 1] * x_input[:]
        return y_predict

    # can use raw x_array to predict y_mean
    def mean_predict(self, x_input):

        tis = self.find_index(x_input)

        y_pn = np.zeros(shape=[len(x_input)])
        for i in range(len(tis)):
            x = x_input[i]
            ti = tis[i]
            x_mean = self.x_mean[ti]
            weights = self.weights[ti]
            try:
                if x >= x_mean:
                    # assert ti + 1 < self.number
                    if ti + 1 >= self.number:
                        ti -= 2
                    x_p = x
                    y_mean = weights[0] + weights[1] * x_mean
                    x_p_mean = self.x_mean[ti + 1]
                    weights_p = self.weights[ti + 1]
                    y_p_mean = weights_p[0] + weights_p[1] * x_p_mean
                    y_p = (y_p_mean - y_mean) / (x_p_mean - x_mean) * (x_p - x_mean) + y_mean
                    y_pn[i] = y_p
            except AssertionError:
                y_pn[i] = self.predict(np.array([x]))[0]
            try:
                if x < x_mean:
                    # assert ti - 1 >= 0
                    if ti - 1 < 0:
                        ti += 2
                    x_n = x
                    y_mean = weights[0] + weights[1] * x_mean
                    x_n_mean = self.x_mean[ti - 1]
                    weights_n = self.weights[ti - 1]
                    y_n_mean = weights_n[0] + weights_n[1] * x_n_mean
                    y_n = (y_n_mean - y_mean) / (x_n_mean - x_mean) * (x_n - x_mean) + y_mean
                    y_pn[i] = y_n
            except AssertionError:
                y_pn[i] = self.predict(np.array([x]))[0]

        return y_pn

    # can use index find x is classified in every tile
    def sort(self, x_input, y=None):
        if y is not None:
            tile_index = self.find_index(x_input)
            sort_x = []
            sort_y = []
            for tile_index_count in range(self.number):
                x_sort = x_input[np.where(tile_index == tile_index_count)]
                y_sort = y[np.where(tile_index == tile_index_count)]
                sort_x.append(x_sort)
                sort_y.append(y_sort)
            return sort_x, sort_y
        tile_index = self.find_index(x_input)
        sort_x = []
        for tile_index_count in range(self.number):
            x_sort = x_input[np.where(tile_index == tile_index_count)]
            sort_x.append(x_sort)
        return sort_x

    def find_index(self, s):
        interval = (self.domain[1] - self.domain[0]) / self.number
        index = (s - self.domain[0]) // interval
        index = index.astype(int)
        for i, v in enumerate(index):
            assert v <= self.number, 'index must smaller than self.number'
            if v == self.number:
                index[i] -= 1

        return index

    # use raw y_array and x_array to gradient descent, alpha is learning rates
    def fit(self, x, y, alpha):
        def partial_w(y_predict, y_true, x_input):
            return 2.0 * (y_predict - y_true) * x_input

        def partial_b(y_predict, y_true):
            return 2.0 * (y_predict - y_true)

        tis = self.find_index(x)
        weights = self.weights
        y_predict = weights[tis, 0] + weights[tis, 1] * x[:]

        for i in range(len(tis)):
            ti = tis[i]
            weights[ti, 0] = weights[ti, 0] - alpha * partial_b(y_predict[i], y[i])
            weights[ti, 1] = weights[ti, 1] - 3 * alpha * partial_w(y_predict[i], y[i], x[i])
        print('loss', np.mean((y_predict - y) ** 2))


# use weights and can use mean predict
class TilePlane:
    def __init__(self, domain=[[-1.0, 1.0], [1.0, 2.0]], number=[10, 20]):
        self.domain1 = domain[0]
        self.domain2 = domain[1]
        self.number1 = number[0]
        self.number2 = number[1]
        self.weights = np.zeros([number[0], number[1], 3])
        self.x_mean = np.zeros([number[0], number[1], 2])

        interval1 = (self.domain1[1] - self.domain1[0]) / self.number1
        interval2 = (self.domain2[1] - self.domain2[0]) / self.number2
        count_i = 0
        for i in range(number[0]):
            count_j = 0
            for j in range(number[1]):
                self.x_mean[i][j][0] = count_i * interval1 + 0.5 * interval1 + self.domain1[0]
                self.x_mean[i][j][1] = count_j * interval2 + 0.5 * interval2 + self.domain2[0]
                count_j += 1
            count_i += 1

    # can use raw x_array to predict y
    def predict(self, x_input1, x_input2):
        if not isinstance(x_input1, Iterable):
            ti1 = self.find_index(x_input1, 0)
            ti2 = self.find_index(x_input2, 1)
            weights = self.weights
            y_predict = weights[ti1, ti2, 0] + weights[ti1, ti2, 1] * x_input1 + weights[ti1, ti2, 2] * x_input2
            return y_predict

        tis1 = self.find_index(x_input1, 0)
        tis2 = self.find_index(x_input2, 1)

        weights = self.weights
        y_predict = weights[tis1, tis2, 0] + weights[tis1, tis2, 1] * x_input1 + weights[tis1, tis2, 2] * x_input2

        return y_predict

    # can use raw x_array to predict y_mean
    def mean_predict(self, x_input1, x_input2):

        if not isinstance(x_input1, Iterable):
            x_input1 = np.array([x_input1])
            x_input2 = np.array([x_input2])
        tis1 = self.find_index(x_input1, 0)
        tis2 = self.find_index(x_input2, 1)

        y_pn = np.zeros(shape=[len(x_input1)])

        for i in range(len(tis1)):
            x1 = x_input1[i]
            x2 = x_input2[i]
            ti1 = tis1[i]
            ti2 = tis2[i]

            xm1 = self.x_mean[ti1, ti2, 0]
            xm2 = self.x_mean[ti1, ti2, 1]
            ym = self.predict(xm1, xm2)
            try:

                # ++
                if x1 >= xm1 and x2 >= xm2:
                    if not (ti1 + 1 < self.number1):
                        ti1 -= 2
                    if not (ti2 + 1 < self.number2):
                        ti2 -= 2
                    # assert ti1 + 1 < self.number1
                    # assert ti2 + 1 < self.number2
                    xdm1 = self.x_mean[ti1 + 1, ti2 + 1, 0]
                    xdm2 = self.x_mean[ti1 + 1, ti2 + 1, 1]

                # +-
                if x1 >= xm1 and x2 < xm2:
                    if not (ti1 + 1 < self.number1):
                        ti1 -= 2
                    if not (ti2 - 1 >= 0):
                        ti2 += 2
                    # assert ti1 + 1 < self.number1
                    # assert ti2 - 1 >= 0
                    xdm1 = self.x_mean[ti1 + 1, ti2 - 1, 0]
                    xdm2 = self.x_mean[ti1 + 1, ti2 - 1, 1]

                # -+
                if x1 < xm1 and x2 >= xm2:
                    if not (ti1 - 1 >= 0):
                        ti1 += 2
                    if not (ti2 + 1 < self.number2):
                        ti2 -= 2
                    # assert ti1 - 1 >= 0
                    # assert ti2 + 1 < self.number2
                    xdm1 = self.x_mean[ti1 - 1, ti2 + 1, 0]
                    xdm2 = self.x_mean[ti1 - 1, ti2 + 1, 1]

                # --
                if x1 < xm1 and x2 < xm2:
                    if not (ti1 - 1 >= 0):
                        ti1 += 2
                    if not (ti2 - 1 >= 0):
                        ti2 += 2
                    # assert ti1 - 1 >= 0
                    # assert ti2 - 1 >= 0
                    xdm1 = self.x_mean[ti1 - 1, ti2 - 1, 0]
                    xdm2 = self.x_mean[ti1 - 1, ti2 - 1, 1]

                p1 = np.array([xm1, xm2, ym])
                p2 = np.array([xdm1, xm2, self.predict(xdm1, xm2)])
                p3 = np.array([xm1, xdm2, self.predict(xm1, xdm2)])

                a = ((p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]))
                b = ((p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2]))
                c = ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                d = (0 - (a * p1[0] + b * p1[1] + c * p1[2]))
                y_pn[i] = (-a * x1 - b * x2 - d) / c

            # out if index_bound
            except AssertionError:
                y_pn[i] = self.predict(x1, x2)

        return y_pn

    # can use index find x is classified in every tile
    def sort(self, x_input1, x_input2, y=None):
        if y is not None:
            tile_index1 = self.find_index(x_input1)
            tile_index2 = self.find_index(x_input2)
            sort_x1 = []
            sort_x2 = []
            sort_y = []
            for tic1 in range(self.number1):
                for tic2 in range(self.number2):
                    index_12 = np.where((tile_index1 == tic1) & (tile_index2 == tic2))
                    x_sort1 = x_input1[index_12]
                    x_sort2 = x_input2[index_12]
                    y_sort = y[index_12]
                    sort_x1.append(x_sort1)
                    sort_x2.append(x_sort2)
                    sort_y.append(y_sort)
            return sort_x1, sort_x2, sort_y
        tile_index1 = self.find_index(x_input1)
        tile_index2 = self.find_index(x_input2)
        sort_x1 = []
        sort_x2 = []

        for tic1 in range(self.number1):
            for tic2 in range(self.number2):
                index_12 = np.where((tile_index1 == tic1) & (tile_index2 == tic2))
                x_sort1 = x_input1[index_12]
                x_sort2 = x_input2[index_12]
                sort_x1.append(x_sort1)
                sort_x2.append(x_sort2)

        return sort_x1, sort_x2

    # input x tell on which tile
    def find_index(self, s, axis):
        assert axis == 0 or axis == 1
        if axis == 0:
            domain = self.domain1
            number = self.number1
        if axis == 1:
            domain = self.domain2
            number = self.number2

        if isinstance(s, Iterable):
            interval = (domain[1] - domain[0]) / number
            index = (s - domain[0]) // interval
            index = index.astype(int)
            for i, v in enumerate(index):
                assert v <= number, 'index must smaller than self.number'
                if v == number:
                    index[i] -= 1
            return index
        else:
            interval = (domain[1] - domain[0]) / number
            index = (s - domain[0]) // interval
            index = int(index)
            assert index <= number, 'index must smaller than self.number'
            if index == number:
                index -= 1
            return index

    # input x1,x2 tell on which tile
    def find_index_2d(self, s1, s2):
        tis1 = self.find_index(s1, 0)
        tis2 = self.find_index(s2, 1)
        return tis1, tis2

    # use raw y_array and x_array to gradient descent, alpha is learning rates
    def fit(self, x1, x2, y, alpha):
        def partial_w(y_predict, y_true, x_input):
            return 2.0 * (y_predict - y_true) * x_input

        def partial_b(y_predict, y_true):
            return 2.0 * (y_predict - y_true)

        tis1, tis2 = self.find_index_2d(x1, x2)
        weights = self.weights
        y_predict = weights[tis1, tis2, 0] + weights[tis1, tis2, 1] * x1 + weights[tis1, tis2, 2] * x2

        for i in range(len(tis1)):
            ti1 = tis1[i]
            ti2 = tis2[i]
            w = weights
            w[ti1, ti2, 0] = w[ti1, ti2, 0] - alpha * partial_b(y_predict[i], y[i])
            w[ti1, ti2, 1] = w[ti1, ti2, 1] - alpha * partial_w(y_predict[i], y[i], x1[i])
            w[ti1, ti2, 2] = w[ti1, ti2, 2] - alpha * partial_w(y_predict[i], y[i], x2[i])


# use y_mean
class TileLine2:
    def __init__(self, domain=[-1.0, 1.0], number=10):
        self.domain = domain
        self.number = number
        self.x_mean = np.zeros([number, 1])
        self.y_mean = np.zeros([number, 1])

        interval = (self.domain[1] - self.domain[0]) / self.number
        count = 0
        for i in range(number):
            self.x_mean[i] = count * interval + 0.5 * interval + self.domain[0]
            count += 1

    # can use raw x_array to predict y_mean
    def mean_predict(self, x_input):

        tis = self.find_index(x_input)

        y_pn = np.zeros(shape=[len(x_input)])
        for i in range(len(tis)):
            x = x_input[i]
            ti = tis[i]
            x_mean = self.x_mean[ti]
            y_mean = self.y_mean[ti]

            if x >= x_mean:

                if ti + 1 >= self.number:
                    ti -= 2
                assert ti + 1 < self.number
                x_p = x
                x_p_mean = self.x_mean[ti + 1]
                y_p_mean = self.y_mean[ti + 1]
                y_p = (y_p_mean - y_mean) / (x_p_mean - x_mean) * (x_p - x_mean) + y_mean
                y_pn[i] = y_p

            if x < x_mean:
                if ti - 1 < 0:
                    ti += 2
                assert ti - 1 >= 0
                x_n = x
                x_n_mean = self.x_mean[ti - 1]
                y_n_mean = self.y_mean[ti - 1]
                y_n = (y_n_mean - y_mean) / (x_n_mean - x_mean) * (x_n - x_mean) + y_mean
                y_pn[i] = y_n

        return y_pn

    # can use index find x is classified in every tile
    def sort(self, x_input, y=None):
        if y is not None:
            tile_index = self.find_index(x_input)
            sort_x = []
            sort_y = []
            for tile_index_count in range(self.number):
                x_sort = x_input[np.where(tile_index == tile_index_count)]
                y_sort = y[np.where(tile_index == tile_index_count)]
                sort_x.append(x_sort)
                sort_y.append(y_sort)
            return sort_x, sort_y
        tile_index = self.find_index(x_input)
        sort_x = []
        for tile_index_count in range(self.number):
            x_sort = x_input[np.where(tile_index == tile_index_count)]
            sort_x.append(x_sort)
        return sort_x

    def find_index(self, s):
        interval = (self.domain[1] - self.domain[0]) / self.number
        index = (s - self.domain[0]) // interval
        index = index.astype(int)
        for i, v in enumerate(index):
            assert v <= self.number, 'index must smaller than self.number'
            if v == self.number:
                index[i] -= 1

        return index

    # use raw y_array and x_array to gradient descent, alpha is learning rates
    def fit(self, x, y, fit_learning_rates):

        tis = self.find_index(x)
        y_mean = self.y_mean

        for i in range(len(tis)):
            ti = tis[i]
            y_mean[ti, 0] = y_mean[ti, 0] + fit_learning_rates * (y[i] - y_mean[ti, 0])


# use y_mean
class TilePlane2:
    def __init__(self, domain=[[-1.0, 1.0], [1.0, 2.0]], number=[10, 20]):
        self.domain1 = domain[0]
        self.domain2 = domain[1]
        self.number1 = number[0]
        self.number2 = number[1]
        self.x_mean = np.zeros([number[0], number[1], 2])
        self.y_mean = np.zeros([number[0], number[1], 1])

        interval1 = (self.domain1[1] - self.domain1[0]) / self.number1
        interval2 = (self.domain2[1] - self.domain2[0]) / self.number2
        count_i = 0
        for i in range(number[0]):
            count_j = 0
            for j in range(number[1]):
                self.x_mean[i][j][0] = count_i * interval1 + 0.5 * interval1 + self.domain1[0]
                self.x_mean[i][j][1] = count_j * interval2 + 0.5 * interval2 + self.domain2[0]
                count_j += 1
            count_i += 1

    # can use raw x_array to predict y_mean
    def mean_predict(self, x_input1, x_input2):

        if not isinstance(x_input1, Iterable):
            x_input1 = np.array([x_input1])
            x_input2 = np.array([x_input2])
        tis1 = self.find_index(x_input1, 0)
        tis2 = self.find_index(x_input2, 1)

        y_pn = np.zeros(shape=[len(x_input1)])

        for i in range(len(tis1)):
            x1 = x_input1[i]
            x2 = x_input2[i]
            ti1 = tis1[i]
            ti2 = tis2[i]

            xm1 = self.x_mean[ti1, ti2, 0]
            xm2 = self.x_mean[ti1, ti2, 1]
            ym = self.y_mean[ti1, ti2, 0]

            # ++
            tix_fix = 0
            tiy_fix = 0
            if x1 >= xm1 and x2 >= xm2:
                if not (ti1 + 1 < self.number1):
                    tix_fix = 2
                if not (ti2 + 1 < self.number2):
                    tiy_fix = 2
                assert ti1 + 1 - tix_fix < self.number1
                assert ti2 + 1 - tiy_fix < self.number2
                x1dm1 = self.x_mean[ti1 + 1 - tix_fix, ti2, 0]
                x2dm1 = self.x_mean[ti1 + 1 - tix_fix, ti2, 1]
                ydm1 = self.y_mean[ti1 + 1 - tix_fix, ti2, 0]

                x1dm2 = self.x_mean[ti1, ti2 + 1 - tiy_fix, 0]
                x2dm2 = self.x_mean[ti1, ti2 + 1 - tiy_fix, 1]
                ydm2 = self.y_mean[ti1, ti2 + 1 - tiy_fix, 0]

            # +-
            if x1 >= xm1 and x2 < xm2:
                if not (ti1 + 1 < self.number1):
                    tix_fix = 2
                if not (ti2 - 1 >= 0):
                    tiy_fix = 2
                assert ti1 + 1 - tix_fix < self.number1
                assert ti2 - 1 + tiy_fix >= 0
                x1dm1 = self.x_mean[ti1 + 1 - tix_fix, ti2, 0]
                x2dm1 = self.x_mean[ti1 + 1 - tix_fix, ti2, 1]
                ydm1 = self.y_mean[ti1 + 1 - tix_fix, ti2, 0]

                x1dm2 = self.x_mean[ti1, ti2 - 1 + tiy_fix, 0]
                x2dm2 = self.x_mean[ti1, ti2 - 1 + tiy_fix, 1]
                ydm2 = self.y_mean[ti1, ti2 - 1 + tiy_fix, 0]

            # -+
            if x1 < xm1 and x2 >= xm2:
                if not (ti1 - 1 >= 0):
                    tix_fix = 2
                if not (ti2 + 1 < self.number2):
                    tiy_fix = 2
                assert ti1 - 1 + tix_fix >= 0
                assert ti2 + 1 - tiy_fix < self.number2
                x1dm1 = self.x_mean[ti1 - 1 + tix_fix, ti2, 0]
                x2dm1 = self.x_mean[ti1 - 1 + tix_fix, ti2, 1]
                ydm1 = self.y_mean[ti1 - 1 + tix_fix, ti2, 0]

                x1dm2 = self.x_mean[ti1, ti2 + 1 - tiy_fix, 0]
                x2dm2 = self.x_mean[ti1, ti2 + 1 - tiy_fix, 1]
                ydm2 = self.y_mean[ti1, ti2 + 1 - tiy_fix, 0]

            # --
            if x1 < xm1 and x2 < xm2:
                if not (ti1 - 1 >= 0):
                    tix_fix = 2
                if not (ti2 - 1 >= 0):
                    tiy_fix = 2
                assert ti1 - 1 + tix_fix >= 0
                assert ti2 - 1 + tiy_fix >= 0
                x1dm1 = self.x_mean[ti1 - 1 + tix_fix, ti2, 0]
                x2dm1 = self.x_mean[ti1 - 1 + tix_fix, ti2, 1]
                ydm1 = self.y_mean[ti1 - 1 + tix_fix, ti2, 0]

                x1dm2 = self.x_mean[ti1, ti2 - 1 + tiy_fix, 0]
                x2dm2 = self.x_mean[ti1, ti2 - 1 + tiy_fix, 1]
                ydm2 = self.y_mean[ti1, ti2 - 1 + tiy_fix, 0]

            p1 = np.array([x1dm1, x2dm1, ydm1])
            p2 = np.array([x1dm2, x2dm2, ydm2])
            p3 = np.array([xm1, xm2, ym])

            a = ((p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]))
            b = ((p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2]))
            c = ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
            d = (0 - (a * p1[0] + b * p1[1] + c * p1[2]))
            y_pn[i] = (-a * x1 - b * x2 - d) / c



        return y_pn

    # can use index find x is classified in every tile
    def sort(self, x_input1, x_input2, y=None):
        if y is not None:
            tile_index1 = self.find_index(x_input1)
            tile_index2 = self.find_index(x_input2)
            sort_x1 = []
            sort_x2 = []
            sort_y = []
            for tic1 in range(self.number1):
                for tic2 in range(self.number2):
                    index_12 = np.where((tile_index1 == tic1) & (tile_index2 == tic2))
                    x_sort1 = x_input1[index_12]
                    x_sort2 = x_input2[index_12]
                    y_sort = y[index_12]
                    sort_x1.append(x_sort1)
                    sort_x2.append(x_sort2)
                    sort_y.append(y_sort)
            return sort_x1, sort_x2, sort_y
        tile_index1 = self.find_index(x_input1)
        tile_index2 = self.find_index(x_input2)
        sort_x1 = []
        sort_x2 = []

        for tic1 in range(self.number1):
            for tic2 in range(self.number2):
                index_12 = np.where((tile_index1 == tic1) & (tile_index2 == tic2))
                x_sort1 = x_input1[index_12]
                x_sort2 = x_input2[index_12]
                sort_x1.append(x_sort1)
                sort_x2.append(x_sort2)

        return sort_x1, sort_x2

    # input x tell on which tile
    def find_index(self, s, axis):
        assert axis == 0 or axis == 1
        if axis == 0:
            domain = self.domain1
            number = self.number1
        if axis == 1:
            domain = self.domain2
            number = self.number2

        if isinstance(s, Iterable):
            interval = (domain[1] - domain[0]) / number
            index = (s - domain[0]) // interval
            index = index.astype(int)
            for i, v in enumerate(index):
                assert v <= number, 'index must smaller than self.number'
                if v == number:
                    index[i] -= 1
            return index
        else:
            interval = (domain[1] - domain[0]) / number
            index = (s - domain[0]) // interval
            index = int(index)
            assert index <= number, 'index must smaller than self.number'
            if index == number:
                index -= 1
            return index

    # input x1,x2 tell on which tile
    def find_index_2d(self, s1, s2):
        tis1 = self.find_index(s1, 0)
        tis2 = self.find_index(s2, 1)
        return tis1, tis2

    # use raw y_array and x_array to gradient descent, alpha is learning rates
    def fit(self, x1, x2, y, fit_learning_rates):

        tis1, tis2 = self.find_index_2d(x1, x2)
        y_mean = self.y_mean

        for i in range(len(tis1)):
            ti1 = tis1[i]
            ti2 = tis2[i]

            y_mean[ti1, ti2, 0] = y_mean[ti1, ti2, 0] + fit_learning_rates * (y[i] - y_mean[ti1, ti2, 0])
