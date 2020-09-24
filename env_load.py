import gym
from brain import Brain
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

env = gym.make('MountainCar-v0')
agent = Brain(action_n=3, state_n=2, tile_num=50)

i_episode = 0

agent.q.tile[0].weights = np.load('./save/W/W{}a0.npy'.format(i_episode), allow_pickle=True)
agent.q.tile[1].weights = np.load('./save/W/W{}a1.npy'.format(i_episode), allow_pickle=True)
agent.q.tile[2].weights = np.load('./save/W/W{}a2.npy'.format(i_episode), allow_pickle=True)

count = -1
RENDER = True
for i_episode in range(99999999):
    observation = env.reset()
    for t in range(10000):
        count += 1
        if t % 1 == 0:
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



env.close()