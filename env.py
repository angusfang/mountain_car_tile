import gym
from brain import Brain
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

env = gym.make('MountainCar-v0')
agent = Brain(3)
count = -1
RENDER = True
for i_episode in range(99999999):
    observation = env.reset()
    for t in range(10000):
        count += 1

        if RENDER:
            env.render()

        s = [np.array([observation[0]]), np.array([observation[1]])]
        a = agent.pi(s)

        observation, reward, done, info = env.step(a)
        if done:

            print("Episode finished after {} timesteps".format(t+1))
            break

        r = reward
        if observation[0] > 0.4:
            print('success')
        s_ = [np.array([observation[0]]), np.array([observation[1]])]
        a_ = agent.pi(s_)

        agent.q.memory_restore(s[0][0], s[1][0], a, r, s_[0][0], s_[1][0], a_)

        if agent.q.memory_index >= agent.q.memory_len:

            agent.q.q_improve()

            # plot
            x1 = np.linspace(-1.2, 0.6, 50)
            x2 = np.linspace(-0.07, 0.07, 50)
            x1, x2 = np.meshgrid(x1, x2)
            x1 = np.reshape(x1, [-1])
            x2 = np.reshape(x2, [-1])

            s_l = []
            a_l = []
            for i in range(x1.shape[0]):
                s = [np.array([x1[i]]), np.array([x2[i]])]
                a = agent.pi(s)
                s_l.append(s)
                a_l.append(a)
            y_l = []
            for i in range(x1.shape[0]):
                y = agent.q.q(s_l[i], a_l[i])
                y_l.append(y)
            y_l = np.array(y_l)

            if 'fig' not in locals().keys():
                fig = plt.figure()
            axis = fig.gca(projection='3d')
            axis.set_xlabel('distance axis')
            axis.set_ylabel('velocity axis')
            axis.set_zlabel('maxQ axis')
            axis.scatter3D(x1, x2, y_l, c=y_l, cmap=cm.coolwarm)
            plt.pause(0.1)
            fig.clf()


env.close()