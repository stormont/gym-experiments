
from agents.dqn import DQNAgent
from algorithms.egreedy import EpsilonGreedyExploration
from algorithms.fixed_q_target import FixedQTarget
from algorithms.loss import huber_loss
from helpers.env_wrapper import EnvironmentWrapper
from algorithms.experience import ExperienceReplay
from algorithms.schedule import ExponentialSchedule
from helpers import data
from helpers.model import ModelWrapper
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import gym
import numpy as np
import sys


def build_network(env, verbose=True):
    model = Sequential()
    model.add(Dense(24, input_dim=env.observation_space.shape[0], activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))
    model.compile(loss=huber_loss, optimizer=Adam(lr=0.001))

    if verbose:
        model.summary()

    return ModelWrapper(model)


def train_dqn(agent, n_episodes=None):
    # Experiment described by: https://github.com/openai/gym/wiki/MountainCarContinuous-v0
    # MountainCar-v0 defines "solving" as getting average reward of -110.0 over 100 consecutive trials.
    # This problem was first described by Andrew Moore in his PhD thesis [Moore90].
    exp_returns = []
    training_complete = False
    e = 0

    # Arbitrary maximum at 2000 episodes, in case of divergent training
    while not training_complete and e < 2000:
        e += 1
        total_reward, n_steps, elapsed_time, ep_reward_est_max, ep_loss_max = agent.train()
        exp_returns.append(total_reward)

        print('Episode {} took {} steps and got {} reward in {} seconds (mean return estimate {}, mean loss {}); epsilon now {}'.format(
            e, n_steps, total_reward, elapsed_time, ep_reward_est_max, ep_loss_max, agent.exploration.epsilon))

        if n_episodes is not None:
            training_complete = e == n_episodes
        else:
            # MountainCar-v0 defines "solving" as getting average reward of -110.0 over 100 consecutive trials.
            training_complete = np.mean(exp_returns[-100:]) >= -110

    print('Training complete after {} episodes'.format(e))
    return exp_returns


def data_exploration(env, n_episodes):
    # Random exploration to establish a baseline
    exp_returns = data.random(env, n_episodes=n_episodes)
    return exp_returns


def basic_dqn(env, n_episodes):
    # Basic DQN with e-greedy exploration
    model = build_network(env)
    decay_sched = ExponentialSchedule(start=1.0, end=0.01, step=0.99)
    exploration = EpsilonGreedyExploration(decay_sched=decay_sched)
    agent = DQNAgent(env, model, gamma=0.99, exploration=exploration)

    # Perform the training
    return train_dqn(agent, n_episodes)


def dqn_with_experience(env, n_episodes):
    # DQN with e-greedy exploration and experience replay
    model = build_network(env)
    experience = ExperienceReplay(maxlen=20000, sample_batch_size=32, min_size_to_sample=1000)
    decay_sched = ExponentialSchedule(start=1.0, end=0.01, step=0.99)
    exploration = EpsilonGreedyExploration(decay_sched=decay_sched)
    agent = DQNAgent(env, model, gamma=0.99, exploration=exploration, experience=experience)

    # Pre-load samples in experience replay.
    # This can also be done implicitly during regular training episodes,
    # but the early training may overfit to early samples.
    experience.bootstrap(env)

    # Perform the training
    return train_dqn(agent, n_episodes)


def dqn_with_fixed_targets(env, n_episodes=None):
    # DQN with e-greedy exploration, experience replay, and fixed-Q targets
    model = build_network(env)
    target_model = build_network(env)
    experience = ExperienceReplay(maxlen=20000, sample_batch_size=32, min_size_to_sample=1000)
    decay_sched = ExponentialSchedule(start=1.0, end=0.01, step=0.99)
    exploration = EpsilonGreedyExploration(decay_sched=decay_sched)
    fixed_target = FixedQTarget(target_model, target_update_step=500, use_soft_targets=True)
    agent = DQNAgent(env, model, gamma=0.99, exploration=exploration, experience=experience, fixed_q_target=fixed_target)

    # Pre-load samples in experience replay.
    # This can also be done implicitly during regular training episodes,
    # but the early training may overfit to early samples.
    experience.bootstrap(env)

    # Perform the training
    return train_dqn(agent, n_episodes)


def dqn_with_prioritized_experience(env, n_episodes=None):
    # DQN with e-greedy exploration, experience replay, and fixed-Q targets
    model = build_network(env)
    target_model = build_network(env)
    experience = ExperienceReplay(maxlen=20000, sample_batch_size=32, min_size_to_sample=1000)
    decay_sched = ExponentialSchedule(start=1.0, end=0.01, step=0.99)
    exploration = EpsilonGreedyExploration(decay_sched=decay_sched)
    fixed_target = FixedQTarget(target_model, target_update_step=500, use_soft_targets=True)
    agent = DQNAgent(env, model, gamma=0.99, exploration=exploration, experience=experience, fixed_q_target=fixed_target)

    # Pre-load samples in experience replay.
    # This can also be done implicitly during regular training episodes,
    # but the early training may overfit to early samples.
    experience.bootstrap(env)

    # Perform the training
    return train_dqn(agent, n_episodes)


def run_single_trials():
    env = EnvironmentWrapper(gym.make('MountainCar-v0'))
    n_episodes = 500

    baseline_returns = data_exploration(env, n_episodes)
    data.report([(baseline_returns, 'b', 'Baseline')], title='Random Walk')

    basic_dqn_returns = basic_dqn(env, n_episodes)
    data.report([(basic_dqn_returns, 'b', 'Basic DQN'),
                 (baseline_returns, 'r', 'Baseline')], title='Vanilla DQN')

    dqn_w_exp_returns = dqn_with_experience(env, n_episodes)
    data.report([(dqn_w_exp_returns, 'b', 'DQN w/ ER'),
                 (baseline_returns, 'r', 'Baseline')], title='Experience Replay')

    dqn_w_fixed_targets_returns = dqn_with_fixed_targets(env, n_episodes)
    data.report([(dqn_w_fixed_targets_returns, 'b', 'DQN w/ Fixed-Q'),
                 (baseline_returns, 'r', 'Baseline')], title='Fixed-Q Targets')

    dqn_w_per_returns = dqn_with_prioritized_experience(env, n_episodes)
    data.report([(dqn_w_per_returns, 'b', 'DQN w/ PER'),
                 (baseline_returns, 'r', 'Baseline')], title='Prioritized ER')

    # Plot all the variations
    data.report([(basic_dqn_returns, 'b', 'Basic DQN'),
                 (dqn_w_exp_returns, 'g', 'DQN w/ ER'),
                 (dqn_w_fixed_targets_returns, 'm', 'DQN w/ Fixed-Q'),
                 (dqn_w_per_returns, 'c', 'DQN w/ PER'),
                 (baseline_returns, 'r', 'Baseline')], title='4 DQN Variants')


def run_multiple_trials():
    env = EnvironmentWrapper(gym.make('MountainCar-v0'))
    n_episodes = 500
    n_trials = 5

    baseline_returns = []

    for i in range(n_trials):
        baseline_returns.append(data_exploration(env, n_episodes))

    data.report([(baseline_returns, 'b', 'Baseline')], title='Random Walk')

    basic_dqn_returns = []

    for i in range(n_trials):
        basic_dqn_returns.append(basic_dqn(env, n_episodes))

    data.report([(basic_dqn_returns, 'b', 'Basic DQN'),
                 (baseline_returns, 'r', 'Baseline')], title='Vanilla DQN')

    dqn_w_exp_returns = []

    for i in range(n_trials):
        dqn_w_exp_returns.append(dqn_with_experience(env, n_episodes))

    data.report([(dqn_w_exp_returns, 'b', 'DQN w/ ER'),
                 (baseline_returns, 'r', 'Baseline')], title='Experience Replay')

    dqn_w_fixed_targets_returns = []

    for i in range(n_trials):
        dqn_w_fixed_targets_returns.append(dqn_with_fixed_targets(env, n_episodes))

    data.report([(dqn_w_fixed_targets_returns, 'b', 'DQN w/ Fixed-Q'),
                 (baseline_returns, 'r', 'Baseline')], title='Fixed-Q Targets')

    dqn_w_per_returns = []

    for i in range(n_trials):
        dqn_w_per_returns.append(dqn_with_prioritized_experience(env, n_episodes))

    data.report([(dqn_w_per_returns, 'b', 'DQN w/ PER'),
                 (baseline_returns, 'r', 'Baseline')], title='Prioritized ER')

    # Plot all the variations
    data.report([(basic_dqn_returns, 'b', 'Basic DQN'),
                 (dqn_w_exp_returns, 'g', 'DQN w/ ER'),
                 (dqn_w_fixed_targets_returns, 'm', 'DQN w/ Fixed-Q'),
                 (dqn_w_per_returns, 'c', 'DQN w/ PER'),
                 (baseline_returns, 'r', 'Baseline')], title='4 DQN Variants')


def solve():
    env = EnvironmentWrapper(gym.make('MountainCar-v0'))
    n_episodes = []

    for i in range(10):
        returns = dqn_with_prioritized_experience(env, n_episodes=None)
        n_episodes.append(len(returns))

    n_episodes = np.array(n_episodes)
    print('MountainCar solved!')
    print('  Median: {} episodes'.format(np.median(n_episodes)))
    print('  Mean:   {} episodes'.format(np.mean(n_episodes)))
    print('  Std:    {} episodes'.format(np.std(n_episodes)))
    print('  Min:    {} episodes'.format(np.min(n_episodes)))
    print('  Max:    {} episodes'.format(np.max(n_episodes)))
    print('  % diverged: {}'.format(len(n_episodes[n_episodes >= 2000]) / float(len(n_episodes))))

    ''' Sample output:
MountainCar solved!
  Median: 309.0 episodes
  Mean:   422.3 episodes
  Std:    241.73624055983004 episodes
  Min:    239 episodes
  Max:    1075 episodes
  % diverged: 0.0
    '''


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
    else:
        arg = 'single'

    if arg == 'multiple':
        run_multiple_trials()
    elif arg == 'solve':
        solve()
    else:
        run_single_trials()


if __name__ == "__main__":
    main()
