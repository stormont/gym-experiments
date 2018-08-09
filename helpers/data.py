
import matplotlib.pyplot as plt
import numpy as np


def _smooth_returns(returns, window=10):
    output = [np.nan] * window

    for i in range(window, len(returns)):
        output.append(np.mean(returns[i-window:i]))

    return output


def _plot_series(series, color, label, smooth_window=10):
    series = np.array(series)

    if series.ndim == 1:
        plt.plot(series, color=color, linewidth=0.5)
        plt.plot(_smooth_returns(series, window=smooth_window), color=color, label=label, linewidth=2)
    else:
        mean = series.mean(axis=0)
        plt.plot(mean, color=color, linewidth=1, label=label)
        plt.fill_between(range(series.shape[1]),
                         mean + series.std(axis=0), mean - series.std(axis=0),
                         color=color, alpha=0.2)


def random(env, n_episodes=1000):
    returns = []

    for _ in range(n_episodes):
        _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = np.random.randint(low=0, high=env.action_space.n)
            _, reward, done, _ = env.step(action)
            total_reward += reward

        returns.append(total_reward)

    return returns


def report(returns, render=True, title=None, legend_loc='upper right', smooth_window=10):
    for i in range(len(returns)):
        series, color, label = returns[i]

        if i == 0:
            print('Experiment stats for {}:'.format(label))
            print('  Mean reward: {}'.format(np.mean(series)))
            print('  Median reward: {}'.format(np.median(series)))
            print('  Std reward: {}'.format(np.std(series)))
            print('  Max reward: {}'.format(np.max(series)))
            print('  Min reward: {}'.format(np.min(series)))

        if not render:
            continue

        _plot_series(series, color=color, label=label, smooth_window=smooth_window)

    if not render:
        return

    if title is not None:
        plt.title(title)

    plt.legend(loc=legend_loc)
    plt.show()
