import gym

if __name__ == '__main__':
  env = gym.make('CartPole-v1')

  # Check out the action and observation spaces
  print("Action space:", env.action_space, "\n")
  print("Observation space:", env.observation_space, "\n")

  # Have to reset the environment once before using it
  # Newer versions of gym will also output an information dictionary, so you'll
  observation = env.reset()
  print("Initial observation:", observation, "\n")

  # Step through the environment once
  random_action = env.action_space.sample()
  observation, reward, done, info = env.step(random_action)
  print("Action taken:", random_action)
  print("Next observation:", observation)
  print("Reward:", reward)
  print("Done?:", done)
  print("Info:", info, "\n")

  # Run and render a 100 steps
  total_reward = 0
  total_steps = 0
  env.reset()
  while total_steps < 100:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    total_reward += reward
    total_steps += 1
    # Reset the environment if the agent dies
    if done:
      obs = env.reset()

  print("Ran %d steps, total reward %.2f" %
        (total_steps, total_reward))
  env.close()
