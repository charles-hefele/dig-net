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

# Print the params
print(f'ENV_NAME: {ENV_NAME}, MAP_NAME: {MAP_NAME}, BATTERY: {BATTERY}, COMPLETION_BONUS: {COMPLETION_BONUS}, '
      f'BATTERY_PENALTY: {BATTERY_PENALTY}')

print(f'SEQUENTIAL_MEMORY: {SEQUENTIAL_MEMORY}, WINDOW_LENGTH: {WINDOW_LENGTH}, STEPS_WARMUP: {STEPS_WARMUP}, '
      f'TARGET_MODEL_UPDATE: {TARGET_MODEL_UPDATE}, LEARNING_RATE: {LEARNING_RATE}, STEPS: {STEPS}')

print(f'PROJECT: {PROJECT}, GROUP: {GROUP}, JOB_TYPE: {JOB_TYPE}')

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME, map_name=MAP_NAME, battery=BATTERY, completion_bonus=COMPLETION_BONUS,
               battery_penalty=BATTERY_PENALTY)
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
memory = SequentialMemory(limit=SEQUENTIAL_MEMORY, window_length=WINDOW_LENGTH)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=STEPS_WARMUP,
               target_model_update=TARGET_MODEL_UPDATE, policy=policy)
dqn.compile(Adam(lr=LEARNING_RATE), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=STEPS, visualize=False, verbose=2, callbacks=[WandbLogger(), FileLogger('results.json')])

# After training is done, we save the final weights.
dqn.save_weights(f'dqn_{ENV_NAME}_{MAP_NAME}_weights.h5f', overwrite=True)

# Render the initial board state
env.reset()
env.render()

# Finally, evaluate our algorithm.
dqn.test(env, nb_episodes=1, visualize=True)

# Terminate wandb
wandb.finish()