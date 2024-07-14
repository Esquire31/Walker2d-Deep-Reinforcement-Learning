import numpy as np
import gymnasium as gym
from agent import ReinforceAgent


class ReinforceTrainer:
    """Train a REINFORCE agent on a given gym environment.
    """

    def __init__(self,
                 env: gym,
                 agent: ReinforceAgent,
                 n_episodes: int,
                 evaluate_interval: int = 100,
                 show_policy_interval: int = 500
                 ):
        """
        Args:
            env (gym.Env): A gym environment
            agent (ReinforceAgent): The REINFORCE agent
            n_episodes (int): Number of episodes to run the environment
            evaluate_interval (int): Number of episodes between two evaluations
            show_policy_interval (int): Number of episodes between policy displays
        """
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.evaluate_interval = evaluate_interval
        self.show_policy_interval = show_policy_interval

    def train(self):
        """Run the training loop.
        """
        # Keep track of the episode lengths and returns
        episode_lengths, episode_returns = [], []

        for episode in range(self.n_episodes + 1):
            curr_episode_length, curr_episode_return = 0, 0
            log_probs, rewards = [], []

            # Run an episode in the environment
            obs, _ = self.env.reset()
            done = False

            while not done:
                action, log_prob = self.agent.get_action(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)
                curr_episode_length += 1
                curr_episode_return += reward

                done = truncated or terminated

                # Update the agent's policy at the end of the episode
            self.agent.update(log_probs, rewards)

            episode_lengths.append(curr_episode_length)
            episode_returns.append(curr_episode_return)

            # Print episode statistics
            if episode % self.evaluate_interval == 0:
                avg_length = np.mean(episode_lengths[-self.evaluate_interval:])
                avg_return = np.mean(episode_returns[-self.evaluate_interval:])
                print(
                    f'Episode {episode}/{self.n_episodes} Average Length: {avg_length:.2f}, Average Return: {avg_return:.2f}')

            if episode % self.show_policy_interval == 0:
                self.show_policy()
        return episode_returns

    def show_policy(self):
        """Show the agent's policy during the training process.
        """
        env = gym.make(self.env.spec.id, render_mode='human')
        obs, _ = env.reset()
        env.render()
        done = False

        while not done:
            action, _ = self.agent.get_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            env.render()
            done = terminated or truncated

        env.close()