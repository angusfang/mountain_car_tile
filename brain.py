import tile_coding
import numpy as np
from collections import Iterable

#  discrete action, 2 continuous state
class Brain:
    def __init__(self, action_n, state_n=2, tile_num=20):
        self.action_n = action_n
        self.greedy = 0.0
        self.q = QApproximate_grad(action_n, state_n=state_n, tile_num=tile_num)


    def pi(self, s):
        if self.greedy < np.random.rand():
            q_value = np.zeros(shape=[self.action_n])
            for a in range(self.action_n):
                q_value[a] = self.q.q(s, a)
            a = np.argmax(q_value)
            bool_arr = (q_value == q_value[a])
            if np.sum(bool_arr) > 1:
                a = np.random.choice(range(self.action_n))
            else:
                pass
        else:
            a = np.random.choice(range(self.action_n))
        return int(a)


class QApproximate:
    def __init__(self, action_n, state_n=2, tile_num=5):
        self.action_n = action_n
        self.tile1 = tile_coding.Tile1D([-1.2, 0.6], tile_num)
        self.tile2 = tile_coding.Tile1D([-0.07, 0.07], tile_num)
        self.weights = np.zeros(shape=[tile_num, tile_num, action_n, state_n + 1])
        # memory: s a r s_ a
        self.memory_len = 100
        self.memory = np.zeros([self.memory_len, state_n + 1 + 1 + state_n + 1])
        self.memory_index = 0

    def memory_restore(self, s1, s2, a, r, s1_, s2_, a_):
        if self.memory_index >= self.memory_len:
            self.memory_index = 0
            self.memory = np.zeros([self.memory_len, 2 + 1 + 1 + 2 + 1])
            print('memory init')
        ratio = self.ratio_test(s1, s2, a)
        if ratio < 0.1:
            self.memory[self.memory_index] = [s1, s2, a, r, s1_, s2_, a_]
            self.memory_index += 1
        else:
            pass

    def q(self, s, a):
        x1 = s[0]
        x2 = s[1]
        tile_d = self.tile1
        tile_v = self.tile2
        tile_index_d = tile_d.x(x1)[1]
        tile_index_v = tile_v.x(x2)[1]
        weights = self.weights
        a = np.array(a)
        a = a.astype(int)
        predicts = weights[tile_index_d, tile_index_v, a, 0] + \
                          weights[tile_index_d, tile_index_v, a, 1] * x1 + \
                          weights[tile_index_d, tile_index_v, a, 2] * x2

        return predicts

    def ratio_test(self, i1, i2, ia):

        x1 = self.memory[:, 0]
        x2 = self.memory[:, 1]
        a = self.memory[:, 2]

        tile_d = self.tile1
        tile_v = self.tile2
        tile_index_d_inp = tile_d.x(np.array([i1]))[1]
        tile_index_v_inp = tile_v.x(np.array([i2]))[1]

        tile_index_d = tile_d.x(x1)[1]
        tile_index_v = tile_v.x(x2)[1]

        d_x_sort = x1[np.where((tile_index_d == tile_index_d_inp) & (tile_index_v == tile_index_v_inp))]
        a_sort = a[np.where((tile_index_d == tile_index_d_inp) & (tile_index_v == tile_index_v_inp))]
        for i in range(self.action_n):
            d_x_sort_a = d_x_sort[np.where(a_sort == i)]

            ratio = len(d_x_sort_a)/self.memory_len

        return ratio

    def q_improve(self):
        alpha = 0.99
        gamma = 0.99
        x1 = self.memory[:, 0]
        x2 = self.memory[:, 1]
        s = [x1, x2]
        a = self.memory[:, 2]
        r = self.memory[:, 3]
        s_ = [self.memory[:, 4], self.memory[:, 5]]
        a_ = self.memory[:, 6]
        q_ = self.q(s, a)+alpha*(r + gamma*self.q(s_, a_) - self.q(s, a))
        tile_d = self.tile1
        tile_v = self.tile2
        tile_index_d = tile_d.x(x1)[1]
        tile_index_v = tile_v.x(x2)[1]

        for tile_index_count_d in range(tile_d.number):
            for tile_index_count_v in range(tile_v.number):
                d_x_sort = x1[np.where((tile_index_d == tile_index_count_d) & (tile_index_v == tile_index_count_v))]
                v_x_sort = x2[np.where((tile_index_d == tile_index_count_d) & (tile_index_v == tile_index_count_v))]
                q_sort = q_[np.where((tile_index_d == tile_index_count_d) & (tile_index_v == tile_index_count_v))]
                a_sort = a[np.where((tile_index_d == tile_index_count_d) & (tile_index_v == tile_index_count_v))]
                for i in range(self.action_n):
                    d_x_sort_a = d_x_sort[np.where(a_sort == i)]
                    v_x_sort_a = v_x_sort[np.where(a_sort == i)]
                    q_sort_a = q_sort[np.where(a_sort == i)]
                    if len(d_x_sort_a) <= 50:
                        continue

                    local_input = np.vstack([d_x_sort_a, v_x_sort_a])
                    local_input = local_input.transpose()

                    d = tile_index_count_d
                    v = tile_index_count_v
                    try:
                        ratio = len(q_sort)/self.memory_len
                        self.weights[d, v, i] = (1 - ratio) * self.weights[d, v, i] + \
                                                            ratio * self.linear_regression(local_input, q_sort_a)
                    except Exception as e:
                        print(e)



    def linear_regression(self, x, y):
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        # if np.linalg.inv(np.matmul(x.T, x)) > 1000 :
        #     return None
        beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)
        return beta

class QApproximate_grad:
    def __init__(self, action_n, state_n=2, tile_num=5, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.action_n = action_n
        self.tile = []
        for i in range(action_n):
            self.tile.append(tile_coding.TilePlane2([[-1.2-0.36, 0.6+0.36], [-0.07-0.028, 0.07+0.028]], [tile_num, tile_num]))

        # memory: s a r s_ a
        self.memory_len = 1
        self.memory = np.zeros([self.memory_len, state_n + 1 + 1 + state_n + 1])
        self.memory_index = 0

    def memory_restore(self, s1, s2, a, r, s1_, s2_, a_):
        if self.memory_index >= self.memory_len:
            self.memory_index = 0
            self.memory = np.zeros([self.memory_len, 2 + 1 + 1 + 2 + 1])
            # print('memory init')
        self.memory[self.memory_index] = [s1, s2, a, r, s1_, s2_, a_]
        self.memory_index += 1
        # ratio = self.ratio_test(s1, s2, a)
        # if ratio < 0.1:
        #     self.memory[self.memory_index] = [s1, s2, a, r, s1_, s2_, a_]
        #     self.memory_index += 1
        # else:
        #     pass

    def q(self, s, a):
        x1 = s[0]
        x2 = s[1]

        if not isinstance(a, Iterable):
            tile = self.tile[a]
            predicts = tile.mean_predict(x1, x2)
        else:
            predicts = np.zeros(shape=[len(a)])
            for i in range(len(a)):
                x1i = np.array([x1[i]])
                x2i = np.array([x2[i]])
                tile = self.tile[int(a[i])]
                predicts[i] = tile.mean_predict(x1i, x2i)[0]

        return predicts

    def ratio_test(self, i1, i2, ia):

        x1 = self.memory[:, 0]
        x2 = self.memory[:, 1]
        a = self.memory[:, 2]

        tile = self.tile[ia]
        tid_input = tile.find_index_2d(i1, i2)
        tid_memorys = tile.find_index_2d(x1, x2)
        tid_12 = np.where((tid_memorys[0] == tid_input[0]) & (tid_memorys[1] == tid_input[1]))

        d_x_sort = x1[tid_12]
        a_sort = a[tid_12]
        for i in range(self.action_n):
            d_x_sort_a = d_x_sort[np.where(a_sort == i)]

            ratio = len(d_x_sort_a)/self.memory_len

        return ratio

    def q_improve(self):
        memory = self.memory[::-1]
        alpha = 1.0
        gamma = 1.0
        x1 = memory[:, 0]
        x2 = memory[:, 1]
        s = [x1, x2]
        a = memory[:, 2]
        r = memory[:, 3]
        s_ = [memory[:, 4], memory[:, 5]]
        a_ = memory[:, 6]
        q_ = self.q(s, a)+alpha*(r + gamma*self.q(s_, a_) - self.q(s, a))

        if not isinstance(a, Iterable):
            tile = self.tile[a]
            tile.fit(s[0], s[1], q_, self.learning_rate)
        else:
            for i in range(len(a)):
                s1i = np.array([s[0][i]])
                s2i = np.array([s[1][i]])
                tile = self.tile[int(a[i])]
                tile.fit(s1i, s2i, q_, self.learning_rate)



    def linear_regression(self, x, y):
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        # if np.linalg.inv(np.matmul(x.T, x)) > 1000 :
        #     return None
        beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)
        return beta