import gym
import gym_digger
import time
from IPython.display import clear_output
import tensorflow as keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(6, input_dim=5, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=150, batch_size=10)

env = gym.make('Digger-v0')

keras.Model()

for episode in range(5):
    env.reset()
    done = False
    step = 0
    total_reward = 0
    while not done:
        step += 1
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        battery = info['battery']
        total_reward += reward

        # print stats
        print(f'episode: {episode}, step: {step}, action: {action}, reward: {reward}, tot_reward: {total_reward}, done: {done}, battery: {battery}')

        # render it
        env.render()

        # delay
        time.sleep(0.05)
        clear_output(wait=True)
