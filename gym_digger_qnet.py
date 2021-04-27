import gym
import gym_digger
import time
from IPython.display import clear_output

env = gym.make('Digger-v0')

for ep in range(100):
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        nutrients = observation[0]
        x_pos = observation[1]
        y_pos = observation[2]
        battery = info['battery']
        print(f'battery: {battery}')

        env.render()
        time.sleep(0.05)
        clear_output(wait=True)