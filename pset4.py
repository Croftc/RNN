from os import system
from time import sleep
import gym
import numpy as np
import pprint
import math
import random

pp = pprint.PrettyPrinter(indent=4)

State = int
Action = int


# NOTE: When testing locally with the `display=True` option
# use `cls` if you are on a Windows machine or `clear` oif you are on an Apple machine.


class DynamicProgramming:
    """
    Write your algorithms for Policy Iteration and Value Iteration in the appropriate functions below.
    """

    def __init__(self, env, gamma=0.95, epsilon=0.1):
        '''
        Initialize policy, environment, value table (V), policy (policy), and transition matrix (P)
        '''
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.policy = self.create_initial_policy()
        self.V = np.zeros(self.num_states) # value function
        self.P = self.env.P # two dimensional array of transition probabilities based on state and action
        self.gamma = gamma
        self.epsilon = epsilon

    def create_initial_policy(self):
        '''
        A policy is a numpy array of length self.num_states where
        self.policy[state] = action

        You are welcome to modify this function to test out the performance of different policies.
        '''
        # policy is num_states array (deterministic)
        policy = np.zeros(self.num_states, dtype=int)
        return policy

    def updated_action_values(self, state: State) -> np.ndarray:
        """
        This is a useful helper function for implementing value_iteration.
        Given a state (given by index), returns a numpy array

            [Q[s, a_1], Q[s, a_2], ..., Q[s, a_n]]

        based on current value function self.V.
        """

        actions = list(self.P[state].keys())
        action_values = []

        # iterate over all actions available in this state
        for action in actions:

            # get potential next states from this action
            potential_states = self.P[state][action]
            action_sum = 0

            # unpack tuple of:
            ## probability of winding up in this next_state
            ## the next state we could be in
            ## the reward from entering this state
            ## whether we're at the goal
            for probability, new_state, reward, is_goal in potential_states:

                # the utility for this state/action is the sum of:
                ## probability winding up in this state * (reward for this state + (gamma * utility of new state))
                action_sum += probability * (reward + (self.gamma * self.V[new_state]))
            # append this sum to our action_values list

            action_values.append(action_sum)
        return action_values

    '''
    # Method to perform value iteration to calculate utilities
    '''
    def value_iteration(self):
        """
        Perform value iteration to compute the value of every state under the optimal policy.
        This method does not return anything. After calling this method, self.V should contain the
        correct values for each state. Additionally, self.policy should be an array that contains
        the optimal policy, where policies are encoded as indicated in the `create_initial_policy` docstring.
        """
        max_delta = math.inf

        # while we're still making non-negligible updates to the utilities
        while max_delta >= (self.epsilon * (1-self.gamma)/self.gamma):
            new_v = []
            deltas = []

            # for each state in the grid
            for state_i in range(self.num_states):

                # get the expected values for all the actions
                action_values = self.updated_action_values(state_i)

                # get the value of the best action
                best_action_value = max(action_values)

                # check how much our utility changed, add it to deltas list
                deltas.append(abs(best_action_value - self.V[state_i]))

                # add this to our new utilities list
                new_v.append(best_action_value)

            # get the biggest change we made
            max_delta = max(deltas)

            # update utilities list
            self.V = new_v

        # update policy based on calculated utilities
        self.update_policy()

    '''
    # Method to update the policy based on calculated utilities
    '''
    def update_policy(self):
        policy = []
        for state_i in range(self.num_states):
            actions = list(self.P[state_i].keys())
            utilities = []
            for action in actions:
                potential_states = self.P[state_i][action]
                utility = 0
                for probability, new_state, reward, is_goal in potential_states:
                    utility += probability * (reward + (self.gamma * self.V[new_state]))
                utilities.append(utility)
            self.policy[state_i] = np.argmax(utilities)
        return policy

    def play_game(self, display=True):
        '''
        Play through one episode of the game under the current policy
        display=True results in displaying the current policy performed on a randomly generated environment in the terminal.
        '''
        self.env.reset()
        episodes = []
        finished = False

        curr_state = self.env.s
        total_reward = 0

        while not finished:
            # display current state
            if display:
                system('clear')
                self.env.render()
                sleep(0.1)

            # find next state
            action = self.policy[curr_state]
            curr_state, reward, finished, info = self.env.step(action)
            total_reward += reward
            episodes.append([curr_state, action, reward])

        # display end result
        if display:
            system('clear')
            self.env.render()

        print(f"Total Reward from this run: {total_reward}")
        return episodes

    def compute_episode_rewards(self, num_episodes=100, step_limit=1000):
        '''
        Computes the mean, variance, and maximum of episode reward over num_episodes episodes
        '''
        total_rewards = np.zeros(num_episodes)
        for episode in range(num_episodes):
            self.env.reset()
            finished = False
            num_steps = 0
            curr_state = self.env.s
            while not finished and num_steps < step_limit:
                action = self.policy[curr_state]
                curr_state, reward, finished, info = self.env.step(action)
                total_rewards[episode] += reward
                num_steps += 1

        return np.mean(total_rewards), np.var(total_rewards), np.max(total_rewards)

    def print_rewards_info(self, num_episodes=100, step_limit=1000):
        '''
        Prints information from compute_episode_rewards
        '''
        mean, var, best = self.compute_episode_rewards(num_episodes=num_episodes, step_limit=step_limit)
        print(f"Mean of Episode Rewards: {mean}, Variance of Episode Rewards: {var}, Best Episode Reward: {best}")


class QLearning:
    """
    Write your algorithm for active model-free Q-learning in the appropriate functions below.
    """

    def __init__(self, env, gamma=0.95, epsilon=0.01):
        """
        Initialize policy, environment, and Q table (Q)
        """
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.state_action_counter = np.zeros((self.num_states, self.num_actions))   # keeps track of k_sa
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state: State) -> Action:
        """
        Returns action based on Q-values using the epsilon-greedy exploration strategy
        """
        j = random.uniform(0,1)

        return self.env.action_space.sample() if j <= self.epsilon else int(np.argmax(self.Q[state,:]))


    def q_learning(self, num_episodes=10000, interval=1000, display=False, step_limit=10000):
        """
        Implement the tabular update for the table of Q-values, stored in self.Q
        Boilerplate code of running several episodes and retrieving the (s, a, r, s') transitions has already been done
        for you.
        """
        mean_returns = []
        for e in range(1, num_episodes+1):
            self.env.reset()
            finished = False

            curr_state = self.env.s
            num_steps = 0

            while not finished and num_steps < step_limit:
                # display current state
                if display:
                    system('clear')
                    self.env.render()
                    sleep(1)

                action = self.choose_action(curr_state)
                next_state, reward, finished, info = self.env.step(action)
                self.state_action_counter[curr_state][action] += 1

                # update Q values. Use the alpha schedule given here.
                alpha = min(0.1, 10 / self.state_action_counter[curr_state][action] ** 0.8)

                self.Q[curr_state][action] = (1-alpha) * self.Q[curr_state][action] + \
                                             alpha * (reward + self.gamma * max(self.Q[next_state,:]))

                num_steps += 1
                curr_state = next_state

            # run tests every interval episodes
            if e % interval == 0:
                if self.epsilon > 0.0001:
                    self.epsilon -= 0.004
                else:
                    print('epsilon bottom')
                mean, var, best = self.compute_episode_rewards(num_episodes=100)
                mean_returns.append(mean)

    # averages rewards over a number of episodes
    def compute_episode_rewards(self, num_episodes=100, step_limit=1000):
        '''
        Computes the mean, variance, and maximum of episode reward over num_episodes episodes
        '''
        total_rewards = np.zeros(num_episodes)
        for episode in range(num_episodes):
            self.env.reset()
            finished = False
            num_steps = 0
            curr_state = self.env.s
            while not finished and num_steps < step_limit:
                best_actions = np.argwhere(self.Q[curr_state] == np.amax(self.Q[curr_state])).flatten()
                action = np.random.choice(best_actions)
                curr_state, reward, finished, info = self.env.step(action)
                total_rewards[episode] += reward
                num_steps += 1

        mean, var, best = np.mean(total_rewards), np.var(total_rewards), np.max(total_rewards)
        print(f"Mean of Episode Rewards: {mean}, Variance of Episode Rewards: {var}, Best Episode Reward: {best}")
        return mean, var, best


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    print("Testing Value Iteration...")
    sleep(1)
    my_policy = DynamicProgramming(env, gamma=0.9)
    my_policy.value_iteration()
    my_policy.play_game()
    my_policy.print_rewards_info()
    sleep(1)

    '''env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    print("Testing Q-Learning...")
    sleep(1)

    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    my_policy = QLearning(env, gamma=0.9, epsilon=0.2)
    my_policy.q_learning(num_episodes=100000, display=False)'''
