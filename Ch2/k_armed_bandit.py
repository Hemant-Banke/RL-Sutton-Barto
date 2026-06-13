import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Bandit_Env:
    def __init__(self, n_arms, bandit_args = {}):
        self.n_arms = n_arms
        self.q_true_var = bandit_args.get('q_true_var', 1)
        self.q_true_mean = bandit_args.get('q_true_mean', 1)
        self.reward_var = bandit_args.get('reward_var', 1)
        self.is_non_stationary = bandit_args.get('is_non_stationary', False)

        self.q_true = None
        self.best_action = None

    def generate_true_action_values(self):
        self.q_true += np.random.normal(self.q_true_mean, self.q_true_var, self.n_arms)  # Sample true action values from a normal distribution
        self.best_action = np.argmax(self.q_true)  # Update best action index based on new true values
    
    def get_reward(self, action):
        return np.random.normal(self.q_true[action], self.reward_var)  # Reward with noise
    
    def reset(self):
        self.q_true = np.zeros(self.n_arms) # Reset true action values to zero
        self.best_action = None
        self.generate_true_action_values()  # Regenerate true action values

    def step(self, action):
        if self.is_non_stationary:
            self.generate_true_action_values()  # Regenerate true action values at each step
        reward = self.get_reward(action)
        return {'reward': reward, 'reporting': {'best_action': self.best_action, 'q_true': self.q_true}}


class Bandit_Actor:
    def __init__(self, n_arms, bandit_env, policy_fn = "epsilon_greedy", update_fn = "constant_step_size", policy_args = {}):
        self.n_arms = n_arms
        self.bandit_env = bandit_env
        self.policy_fn = policy_fn
        self.update_fn = update_fn
        self.policy_args = policy_args
        self.q_estimate_initial = policy_args.get('q_estimate_initial', 0)

        self.q_estimate = np.zeros(self.n_arms) + self.q_estimate_initial # Reset estimates
        self.action_count = np.zeros(self.n_arms)  # Reset action counts
        self.total_count = 0  # Total number of actions taken
        self.rewards = None
        self.actions = None
        self.best_action_count = None
        self.best_action_perct = None
        self.mse_history = None
        self.store = {}  # For custom update/policy functions to store state if needed

    def update_action_estimate(self, action, reward):
        if self.update_fn == "sample_average":
            step_size = 1 / self.action_count[action]  # Use sample average method
            self.q_estimate[action] += step_size * (reward - self.q_estimate[action])
        
        elif self.update_fn == "constant_step_size":
            step_size = self.policy_args.get('step_size', 0.1)
            self.q_estimate[action] += step_size * (reward - self.q_estimate[action])

        elif self.update_fn == "constant_step_size_unbiased":
            alpha = self.policy_args.get('step_size', 0.1)
            beta_n = alpha / (1 - (1 - alpha) ** self.action_count[action])  # Unbiased constant step size
            self.q_estimate[action] += beta_n * (reward - self.q_estimate[action])

        elif callable(self.update_fn):
            self.update_fn(self, action, reward)  # Custom update function

        else:
            raise ValueError(f"Unknown update function: {self.update_fn}")
        
    def policy(self):
        noise = np.random.random(self.n_arms) * 1e-5  # Small noise to break ties
        
        if self.policy_fn == 'greedy':
            return np.argmax(self.q_estimate + noise)  # Greedy action with tie-breaking
        
        elif self.policy_fn == 'epsilon_greedy':
            epsilon = self.policy_args.get('epsilon', 0.1)
            if np.random.rand() < epsilon:
                return np.random.randint(self.n_arms)  # Explore
            else:
                return np.argmax(self.q_estimate + noise)  # Exploit
        
        elif self.policy_fn == 'upper_cb':
            confidence = self.policy_args.get('confidence', 2)
            bound_term = 0 if self.total_count == 0 else confidence * np.sqrt(np.log(self.total_count) / (self.action_count + 1e-5))
            upper_bound = self.q_estimate + bound_term  # UCB calculation
            return np.argmax(upper_bound + noise)  # Action with highest upper confidence bound

        elif callable(self.policy_fn):
            return self.policy_fn(self)  # Custom policy function

        else:
            raise ValueError(f"Unknown policy function: {self.policy_fn}")
        
    def take_action(self, action):
        self.action_count[action] += 1
        self.total_count += 1

    def run(self, n_steps):
        self.rewards = np.zeros(n_steps)
        self.actions = np.zeros(n_steps, dtype=int)
        self.best_action_count = np.zeros(n_steps)
        self.mse_history = np.zeros(n_steps)  # To track MSE of estimates over time
        
        self.bandit_env.reset()  # Reset the environment at the start of the run
        for step in range(n_steps):
            # Decide Action based on current state
            action = self.policy()
            self.actions[step] = action

            # Take Action
            self.take_action(action)

            # Observe Reward and Update Estimates
            env_result = self.bandit_env.step(action)
            reward = env_result['reward']
            self.update_action_estimate(action, reward)
            self.rewards[step] = reward
            
            # Reporting
            _best_action = env_result['reporting']['best_action']
            _q_true = env_result['reporting']['q_true']
            if action == _best_action:
                self.best_action_count[step] = 1
            self.mse_history[step] = ((self.q_estimate - _q_true) ** 2).mean()  # MSE of estimates
        self.best_action_perct = np.cumsum(self.best_action_count) / (np.arange(n_steps) + 1)  # Cumulative average of optimal action count



def run_bandit_experiments(actor_map, n_steps, n_iterations):
    rewards_map = {actor: pd.DataFrame(np.zeros((n_iterations, n_steps))) for actor in actor_map}
    best_action_perct_map = {actor: pd.DataFrame(np.zeros((n_iterations, n_steps))) for actor in actor_map}
    mse_map = {actor: pd.DataFrame(np.zeros((n_iterations, n_steps))) for actor in actor_map}

    for i in range(n_iterations):
        for actor in actor_map:
            bandit_actor = Bandit_Actor(**actor_map[actor])
            bandit_actor.run(n_steps)

            rewards_map[actor].iloc[i] = bandit_actor.rewards
            best_action_perct_map[actor].iloc[i] = bandit_actor.best_action_perct
            mse_map[actor].iloc[i] = bandit_actor.mse_history

    return { 'rewards': rewards_map, 'best_action_perct': best_action_perct_map, 'mse': mse_map }



def diagnostic_plots(actor_map, rewards_map, best_action_perct_map, mse_map):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    for actor in actor_map:
        axs[0].plot(rewards_map[actor].mean(), label=actor)
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Reward')
    axs[0].set_title('Bandit Rewards Over Time')
    axs[0].legend()
    axs[0].grid()

    for actor in actor_map:
        axs[1].plot(best_action_perct_map[actor].mean(), label=actor)
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('Optimal Action Percentage')
    axs[1].set_title('Bandit Optimal Action Percentages Over Time')
    axs[1].legend()
    axs[1].grid()

    for actor in actor_map:
        axs[2].plot(mse_map[actor].mean(), label=actor)
    axs[2].set_xlabel('Steps')
    axs[2].set_ylabel('MSE')
    axs[2].set_title('Action Estimation MSE Over Time')
    axs[2].legend()
    axs[2].grid()

    plt.show()
