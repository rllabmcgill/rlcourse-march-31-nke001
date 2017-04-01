import gym
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras import backend as K
import numpy as np


batch_size = 100
hidden_size = 100
layers = 2
batch_norm = False
min_train = 10
train_repeat = 10
gamma = 0.995
tau = 0.001
episodes = 200
activation = 'tanh'
optimizer = 'adam'
optimizer_lr = 0.001
noise_decay = 'linear'
fixed_noise  = 0.1
display = False
environment = 'CartPole-v0'
gym_monitor = False
max_timesteps = 200
exploration = 0.1
advantage = 'naive' #max, average!
env = gym.make(environment)


def createLayers():
  x = Input(shape=env.observation_space.shape)
  h = x
  for i in xrange(layers):
    h = Dense(hidden_size, activation=activation)(h)
  y = Dense(env.action_space.n + 1)(h)
  z = Lambda(lambda a: K.expand_dims(a[:,0], dim=-1) + a[:,1:], output_shape=(env.action_space.n,))(y)
  return x, z

x, z = createLayers()
model = Model(input=x, output=z)
model.summary()
model.compile(optimizer='adam', loss='mse')

x, z = createLayers()
target_model = Model(input=x, output=z)
target_model.set_weights(model.get_weights())

prestates = []
actions = []
rewards = []
poststates = []
terminals = []
episode_reward_list = []
total_reward = 0
for i_episode in xrange(episodes):
    observation = env.reset()
    episode_reward = 0
    for t in xrange(max_timesteps):
        if display:
          env.render()
        if np.random.random() < exploration:
          action = env.action_space.sample()
        else:
          s = np.array([observation])
          q = model.predict(s, batch_size=1)
          action = np.argmax(q[0])

        prestates.append(observation)
        actions.append(action)
        observation, reward, done, info = env.step(action)
        episode_reward += reward

        rewards.append(reward)
        poststates.append(observation)
        terminals.append(done)

        if len(prestates) > min_train:
          for k in xrange(train_repeat):
            if len(prestates) > batch_size:
              indexes = np.random.choice(len(prestates), size=batch_size)
            else:
              indexes = range(len(prestates))

            qpre = model.predict(np.array(prestates)[indexes])
            qpost = model.predict(np.array(poststates)[indexes])
            for i in xrange(len(indexes)):
              if terminals[indexes[i]]:
                qpre[i, actions[indexes[i]]] = rewards[indexes[i]]
              else:
                qpre[i, actions[indexes[i]]] = rewards[indexes[i]] + gamma * np.amax(qpost[i])
            model.train_on_batch(np.array(prestates)[indexes], qpre)
            weights = model.get_weights()
            target_weights = target_model.get_weights()
            for i in xrange(len(weights)):
              target_weights[i] = tau * weights[i] + (1 - tau) * target_weights[i]
            target_model.set_weights(target_weights)

        if done:
            break

    print "Episode {} finished after {} timesteps, episode reward {}".format(i_episode + 1, t + 1, episode_reward)
    total_reward += episode_reward
    episode_reward_list.append(episode_reward)

np.savez('episode_reward_list.npz', episode_reward_list)
print "Average reward per episode {}".format(total_reward / episodes)
A

