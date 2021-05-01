import numpy as np
import gym
import gym_digger

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import WandbLogger, FileLogger
import wandb

ENV_NAME = 'Digger-v0'
MAP_NAME = '2x2_e'

wandb.init(project='digger',
           group=MAP_NAME,
           job_type='6-values',
           settings=wandb.Settings(start_method="thread"))

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME, map_name=MAP_NAME)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2, callbacks=[WandbLogger(), FileLogger('results.json')])

# After training is done, we save the final weights.
dqn.save_weights(f'dqn_{ENV_NAME}_{MAP_NAME}_weights.h5f', overwrite=True)

# Finally, evaluate our algorithm.
dqn.test(env, nb_episodes=1, visualize=True)