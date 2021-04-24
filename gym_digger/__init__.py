from gym.envs.registration import register

register(
    id='Digger-v0',
    entry_point='gym_digger.envs:DiggerEnv',
)

register(
    id='DiggerDiscrete-v0',
    entry_point='gym_digger.envs:DiggerEnvDiscrete',
)
