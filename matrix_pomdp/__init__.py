from gymnasium.envs.registration import register

register(
    id='matrix_pomdp/MatrixPOMDP-v0',
    entry_point='matrix_pomdp.envs:MatrixPOMDPEnv',
    max_episode_steps=300,
)