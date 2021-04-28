import gym
import gym_digger
import time
from IPython.display import clear_output

env = gym.make('Digger-v0')

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
