import gymnasium as gym

env = gym.make('Walker2d-v4', render_mode='human')
print(env.action_space)
print(env.observation_space)

obs, _ = env.reset()
env.render()

done = False
episode_length = 0
total_reward = 0

while not done:
    action = (0,0,0,0,0,0)
    obs, reward, terminated, truncated, _ = env.step(action)
    episode_length += 1
    total_reward += reward
    done = terminated or truncated

print(f'Episode length: {episode_length}, total reward: {total_reward: .2f}')
env.close()