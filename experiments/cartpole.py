
from agents.dqn import DQNAgent
from algorithms.egreedy import ExponentialEpsilonGreedyExploration
from algorithms.fixed_q_target import FixedQTarget
from helpers.env_wrapper import EnvironmentWrapper
from algorithms.experience import ExperienceReplay
from helpers import data
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import gym
from helpers.model import ModelWrapper


def build_network(env, verbose=True):
    model = Sequential()
    model.add(Dense(24, input_dim=env.observation_space.shape[0], activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    if verbose:
        model.summary()

    return ModelWrapper(model)


def train_dqn(agent, n_episodes):
    exp_returns = []

    for e in range(n_episodes):
        total_reward, n_steps, elapsed_time = agent.train()
        exp_returns.append(total_reward)

        print('Episode {} took {} steps and got {} reward in {} seconds; epsilon now {}'.format(
            e+1, n_steps, total_reward, elapsed_time, agent.exploration.epsilon))

    return exp_returns


def data_exploration(env, n_episodes):
    # Random exploration to establish a baseline
    exp_returns = data.random(env, n_episodes=n_episodes)
    return exp_returns


def basic_dqn(env, n_episodes):
    # Basic DQN with e-greedy exploration
    model = build_network(env)
    exploration = ExponentialEpsilonGreedyExploration(epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.99)
    agent = DQNAgent(env, model, gamma=0.99, exploration=exploration)

    # Perform the training
    return train_dqn(agent, n_episodes)


def basic_dqn_with_experience(env, n_episodes):
    # Basic DQN with e-greedy exploration and experience replay
    model = build_network(env)
    experience = ExperienceReplay(maxlen=2000, sample_batch_size=32, min_size_to_sample=100)
    exploration = ExponentialEpsilonGreedyExploration(epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.99)
    agent = DQNAgent(env, model, gamma=0.99, exploration=exploration, experience=experience)

    # Pre-load samples in experience replay.
    # This can also be done implicitly during regular training episodes,
    # but the early training may overfit to early samples.
    experience.bootstrap(env)

    # Perform the training
    return train_dqn(agent, n_episodes)


def basic_dqn_with_fixed_targets(env, n_episodes):
    # Basic DQN with e-greedy exploration, experience replay, and fixed-Q targets
    model = build_network(env)
    target_model = build_network(env)
    experience = ExperienceReplay(maxlen=2000, sample_batch_size=32, min_size_to_sample=100)
    exploration = ExponentialEpsilonGreedyExploration(epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.99)
    fixed_target = FixedQTarget(target_model, target_update_step=500, use_soft_targets=True)
    agent = DQNAgent(env, model, gamma=0.99, exploration=exploration, experience=experience, fixed_q_target=fixed_target)

    # Pre-load samples in experience replay.
    # This can also be done implicitly during regular training episodes,
    # but the early training may overfit to early samples.
    experience.bootstrap(env)

    # Perform the training
    return train_dqn(agent, n_episodes)


def main():
    env = EnvironmentWrapper(gym.make('CartPole-v1'))
    n_episodes = 500

    baseline_returns = data_exploration(env, n_episodes)
    data.report([(baseline_returns, 'r', 'Baseline')])

    basic_dqn_returns = basic_dqn(env, n_episodes)
    data.report([(basic_dqn_returns, 'b', 'Basic DQN'),
                 (baseline_returns, 'r', 'Baseline')])

    basic_dqn_w_exp_returns = basic_dqn_with_experience(env, n_episodes)
    data.report([(basic_dqn_w_exp_returns, 'b', 'DQN w/ ER'),
                 (baseline_returns, 'r', 'Baseline')])

    basic_dqn_w_fixed_targets_returns = basic_dqn_with_fixed_targets(env, n_episodes)
    data.report([(basic_dqn_w_fixed_targets_returns, 'b', 'DQN w/ Fixed-Q'),
                 (baseline_returns, 'r', 'Baseline')])

    # Plot all the variations
    data.report([(basic_dqn_returns, 'b', 'Basic DQN'),
                 (basic_dqn_w_exp_returns, 'g', 'DQN w/ ER'),
                 (basic_dqn_w_fixed_targets_returns, 'm', 'DQN w/ Fixed-Q'),
                 (baseline_returns, 'r', 'Baseline')])


if __name__ == "__main__":
    main()
