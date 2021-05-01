from gym.envs.registration import register

register(
    id='Digger-v0',
    entry_point='gym_digger.envs:DiggerEnv',
    max_episode_steps=100000
)

register(
    id='DiggerDiscrete-v0',
    entry_point='gym_digger.envs:DiggerEnvDiscrete',
)
