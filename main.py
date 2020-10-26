import gym
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

from brain import Brain

env = gym.make('MountainCar-v0')
agent = Brain(action_n=3, state_n=2, tile_num=20)

i_ = 9900
load_loc = 9
save_loc = 9
agent.q.tile[0].y_mean = np.load('./save{}/M/M{}a0.npy'.format(load_loc,i_), allow_pickle=True)
agent.q.tile[1].y_mean = np.load('./save{}/M/M{}a1.npy'.format(load_loc,i_), allow_pickle=True)
agent.q.tile[2].y_mean = np.load('./save{}/M/M{}a2.npy'.format(load_loc,i_), allow_pickle=True)
count = -1
RENDER = True
SUCCESS = False
NORMAL_SAVE = True
for i_episode in range(10000):
    i_episode += i_
    observation = env.reset()
    SUCCESS = False
    NORMAL_SAVE = True
    done = False
    for t in range(10000):
        count += 1

        s = [np.array([observation[0]]), np.array([observation[1]])]
        a = agent.pi(s)

        observation, reward, done, info = env.step(a)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            if t < 100:
                SUCCESS = True
                # r = r + 1

                print('save t < 300 model')
                np.save('./save{}/1_success/{}'.format(save_loc, i_episode), np.array([0]))
                np.save('./save{}/1_success/M{}a0'.format(save_loc, i_episode), agent.q.tile[0].y_mean)
                np.save('./save{}/1_success/M{}a1'.format(save_loc, i_episode), agent.q.tile[1].y_mean)
                np.save('./save{}/1_success/M{}a2'.format(save_loc, i_episode), agent.q.tile[2].y_mean)
            break

        r = reward



        s_ = [np.array([observation[0]]), np.array([observation[1]])]
        a_ = agent.pi(s_)

        agent.q.memory_restore(s[0][0], s[1][0], a, r, s_[0][0], s_[1][0], a_)
        agent.q.q_improve()
        # print('improve 1 times')
    print('i_episode: {}.'.format(i_episode))

    if i_episode % 100 == 0:
        # plot
        x1 = np.linspace(-1.2 - 0.36, 0.6 + 0.36, 100)
        x2 = np.linspace(-0.07 - 0.028, 0.07 + 0.028, 100)
        x1, x2 = np.meshgrid(x1, x2)
        x1 = np.reshape(x1, [-1])
        x2 = np.reshape(x2, [-1])

        s = np.array([x1, x2])
        y_l = []
        y_l0 = []
        y_l1 = []
        y_l2 = []
        for i in range(x1.shape[0]):
            a = agent.pi(s.T[i])
            y = agent.q.q(s.T[i], a)
            y0 = agent.q.q(s.T[i], 0)
            y1 = agent.q.q(s.T[i], 1)
            y2 = agent.q.q(s.T[i], 2)
            y_l.append(y)
            y_l0.append(y0)
            y_l1.append(y1)
            y_l2.append(y2)
        y_l = np.array(y_l)
        y_l0 = np.array(y_l0)
        y_l1 = np.array(y_l1)
        y_l2 = np.array(y_l2)

        fig = plt.figure()


        def axis(fig, label, loc):
            axis = fig.add_subplot(2, 2, loc, projection='3d')
            axis.set_xlabel('distance axis')
            axis.set_ylabel('velocity axis')
            axis.set_zlabel('{} axis'.format(label))
            axis.set_zlim(-500, 500)

            return axis


        axis1 = axis(fig, '-a0_q', 1)
        axis2 = axis(fig, '-a1_q', 2)
        axis3 = axis(fig, '-a2_q', 3)
        axis4 = axis(fig, '-max_q', 4)

        x1 = np.reshape(x1, [100, 100])
        x2 = np.reshape(x2, [100, 100])
        y_l = np.reshape(y_l, [100, 100])
        y_l0 = np.reshape(y_l0, [100, 100])
        y_l1 = np.reshape(y_l1, [100, 100])
        y_l2 = np.reshape(y_l2, [100, 100])
        axis4.plot_surface(x1, x2, -y_l, cmap=cm.coolwarm)
        axis1.plot_surface(x1, x2, -y_l0, cmap='Blues')
        axis2.plot_surface(x1, x2, -y_l1, cmap='Greens')
        axis3.plot_surface(x1, x2, -y_l2, cmap='Oranges')
        # print(np.sum(y_l0 == y_l1))
        # print(np.sum(y_l0 == y_l2))
        # print(np.sum(y_l1 == y_l2))
        fig.suptitle("episode{}".format(i_episode), fontsize=16)
        plt.savefig('./save{}/Q/Q{}.png'.format(save_loc, i_episode))
        plt.close('all')
        if NORMAL_SAVE:
            np.save('./save{}/M/M{}a0'.format(save_loc, i_episode), agent.q.tile[0].y_mean)
            np.save('./save{}/M/M{}a1'.format(save_loc, i_episode), agent.q.tile[1].y_mean)
            np.save('./save{}/M/M{}a2'.format(save_loc, i_episode), agent.q.tile[2].y_mean)
        print("episode{}".format(i_episode))

env.close()
