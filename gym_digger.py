import gym
import gym_digger
from gym import envs

# env = gym.make('Digger-v0')
env = gym.make('DiggerDiscrete-v0')

# print all envs
# envids = [spec.id for spec in envs.registry.all()]
# for envid in sorted(envids):
#     print(envid)

for step in range(300):
    print(f'step: {step}')
    print(f'state: {env.s}')
    # new_state, reward, done, info = env.step(4)
    env.render()
    # env.step(1)
    # env.render()
    # env.step(2)
    # env.render()
    # env.step(3)
    # env.render()
    # env.step(4)
    # env.render()