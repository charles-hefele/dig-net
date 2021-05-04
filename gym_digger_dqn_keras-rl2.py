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

# environment settings
ENV_NAME = 'Digger-v0'
MAP_NAME = '2x2_e'
BATTERY = 200
COMPLETION_BONUS = 100
BATTERY_PENALTY = 0
OUT_FILE = f'dqn_{ENV_NAME}_{MAP_NAME}_weights.h5f'

# keras settings
SEQUENTIAL_MEMORY = 50000
WINDOW_LENGTH = 1
STEPS_WARMUP = 10
TARGET_MODEL_UPDATE = 1e-2
LEARNING_RATE = 1e-3
STEPS = 50000

# wandb settings
PROJECT = ENV_NAME
GROUP = MAP_NAME
JOB_TYPE = '6-values'

# start wandb
wandb.init(project=PROJECT,
           group=MAP_NAME,
           job_type=JOB_TYPE,
           settings=wandb.Settings(start_method="thread"))

# print the params
print(f'ENV_NAME: {ENV_NAME}, MAP_NAME: {MAP_NAME}, BATTERY: {BATTERY}, COMPLETION_BONUS: {COMPLETION_BONUS}, '
      f'BATTERY_PENALTY: {BATTERY_PENALTY}')

print(f'SEQUENTIAL_MEMORY: {SEQUENTIAL_MEMORY}, WINDOW_LENGTH: {WINDOW_LENGTH}, STEPS_WARMUP: {STEPS_WARMUP}, '
      f'TARGET_MODEL_UPDATE: {TARGET_MODEL_UPDATE}, LEARNING_RATE: {LEARNING_RATE}, STEPS: {STEPS}')

print(f'PROJECT: {PROJECT}, GROUP: {GROUP}, JOB_TYPE: {JOB_TYPE}')

# build the environment
env = gym.make(ENV_NAME, map_name=MAP_NAME, battery=BATTERY, completion_bonus=COMPLETION_BONUS,
               battery_penalty=BATTERY_PENALTY)
nb_actions = env.action_space.n

# build the model
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

# configure and compile the agent
memory = SequentialMemory(limit=SEQUENTIAL_MEMORY, window_length=WINDOW_LENGTH)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=STEPS_WARMUP,
               target_model_update=TARGET_MODEL_UPDATE, policy=policy)
dqn.compile(Adam(lr=LEARNING_RATE), metrics=['mae'])

# train the agent
dqn.fit(env, nb_steps=STEPS, visualize=False, verbose=2, callbacks=[WandbLogger(), FileLogger('results.json')])

# save the weights
dqn.save_weights(OUT_FILE, overwrite=True)

# render the initial environment state
print('Initial board state', end='')
env.reset()
env.render()

# test the agent
dqn.test(env, nb_episodes=1, visualize=True)

# terminate wandb
wandb.finish()
