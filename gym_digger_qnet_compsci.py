import gym
import gym_digger
import numpy as np
import random
from gym import envs
from IPython.display import clear_output
import time
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from IPython import display
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = 12, 8

env = gym.make('Digger-v0')

class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        self.observation_space = env.observation_space
        print('Action size:', self.action_size)
        print('Observation space:', self.observation_space)

    def get_action(self):
        return random.choice(range(self.action_size))

class QNAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
        super().__init__(env)
        # self.state_size = env.observation_space.n
        # print('State size:', self.state_size)

        self.epsilon = 1.0
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.build_model()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        tf.reset_default_graph()
        # self.state_in = tf.placeholder(tf.int32, shape=[2])
        self.action_in = tf.placeholder(tf.int32, shape=[1])
        self.target_in = tf.placeholder(tf.float32, shape=[1])

        # self.state = tf.one_hot(self.state_in, depth=self.state_size)
        # self.state = tf.size(self.observation_space.shape)

        self.action = tf.one_hot(self.action_in, depth=self.action_size)

        self.q_state = tf.layers.dense(self.observation_space.shape, units=self.action_size, name='q_table')
        self.q_action = tf.reduce_sum(tf.multiply(self.q_state, self.action), axis=1)

        self.loss = tf.reduce_sum(tf.square(self.target_in - self.q_action))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def get_action(self, state):
        q_state = self.sess.run(self.q_state, feed_dict={self.state_in: [state]})
        action_greedy = np.argmax(q_state)
        action_random = super().get_action()
        return action_random if random.random() < self.epsilon else action_greedy

    def train(self, experience):
        state, action, next_state, reward, done = ([exp] for exp in experience)

        q_next = self.sess.run(self.q_state, feed_dict={self.state_in: next_state})
        q_next[done] = np.zeros([self.action_size])
        q_target = reward + self.discount_rate * np.max(q_next)

        feed = {self.state_in: state, self.action_in: action, self.target_in: q_target}
        self.sess.run(self.optimizer, feed_dict=feed)

        if experience[4]:
            self.epsilon = self.epsilon * 0.99

    def __del__(self):
        self.sess.close()

agent = QNAgent(env)

# training
total_reward = 0
all_rewards = []
for episode in range(500):
    state = env.reset()
    done = False
    steps = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        steps += 1
        agent.train((state, action, next_state, reward, done))
        state = next_state
        total_reward += reward

        #         print('s:', state, 'a:', action)
        #         print('Episode: {}, Total reward: {}, Steps: {}, epsilon: {:.2f}'.format(episode, total_reward, steps, agent.epsilon))
        #         env.render()
        with tf.variable_scope('q_table', reuse=True):
            weights = agent.sess.run(tf.get_variable('kernel'))
        #             print(weights)
    #         time.sleep(0.25)
    #         clear_output(wait=True)
    print('Episode: {}, Total reward: {}, Steps: {}, epsilon: {:.2f}'.format(episode, total_reward, steps, agent.epsilon))
    all_rewards.append(total_reward)
plt.plot(all_rewards)

# testing
total_reward = 0
agent.epsilon = 0
for episode in range(10):
    state = env.reset()
    done = False
    steps = 0
    while not done:
        # update
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        steps += 1
        state = next_state
        total_reward += reward

        # output
        print('s:', state, 'a:', action)
        print('Episode: {}, Total reward: {}, Steps: {}, epsilon: {:.2f}'.format(episode, total_reward, steps, agent.epsilon))
        env.render()
        time.sleep(0.2)
        clear_output(wait=True)