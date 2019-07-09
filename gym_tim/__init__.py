from gym.envs.registration import register

register(
    id='tim_pendulum-v0',
    entry_point='gym_tim.envs:TimPendWrap',
)

register(
    id='tim_magman-v0',
    entry_point='gym_tim.envs:TimMagWrap',
)
