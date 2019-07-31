import numpy as np
import matplotlib.pyplot as plt
import pickle


def running_mean(x):
	a = [x[0]]
	for i in range(1, len(x)):
		a.append(a[i - 1] * .99 + x[i] * .01)
	return a

episodes = 10000

# scores = pickle.load(open('results/scores_dqn.pkl', 'rb'))
# rewards = pickle.load(open('results/rewards_variation_dqn.pkl', 'rb'))
# steps = pickle.load(open('results/steps_variation_dqn.pkl', 'rb'))

# scores = pickle.load(open('results/scores_dqn_exp.pkl', 'rb'))
# rewards = pickle.load(open('results/rewards_variation_dqn_exp.pkl', 'rb'))
# steps = pickle.load(open('results/steps_variation_dqn_exp.pkl', 'rb'))

# scores = pickle.load(open('results/scores_dqn_double.pkl', 'rb'))
# rewards = pickle.load(open('results/rewards_variation_dqn_double.pkl', 'rb'))
# steps = pickle.load(open('results/steps_variation_dqn_double.pkl', 'rb'))

# scores = pickle.load(open('results/scores_dqn_double_exp.pkl', 'rb'))
# rewards = pickle.load(open('results/rewards_variation_dqn_double_exp.pkl', 'rb'))
# steps = pickle.load(open('results/steps_variation_dqn_double_exp.pkl', 'rb'))

scores = pickle.load(open('results/scores_random.pkl', 'rb'))
rewards = pickle.load(open('results/rewards_variation_random.pkl', 'rb'))
steps = pickle.load(open('results/steps_variation_random.pkl', 'rb'))


fig = plt.figure()
scoreplot = fig.add_subplot(311)
rewardsplot = fig.add_subplot(312)
stepsplot = fig.add_subplot(313)

scoreplot.plot(np.arange(1, episodes + 1), running_mean(scores))
scoreplot.set_xlabel('Episode')
scoreplot.set_ylabel('Score')
scoreplot.grid(True)

rewardsplot.plot(np.arange(1, episodes + 1), running_mean(rewards))
rewardsplot.set_xlabel('Episode')
rewardsplot.set_ylabel('Rewards Variation')
rewardsplot.grid(True)

stepsplot.plot(np.arange(1, episodes + 1), running_mean(steps))
stepsplot.set_xlabel('Episode')
stepsplot.set_ylabel('Steps Variation')
stepsplot.grid(True)

plt.show()