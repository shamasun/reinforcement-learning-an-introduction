import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# A class that selects an action --> gets rewarded --> re-estimates action-values --> selects an action
class Bandit:
    def __init__(self, k=10, epsilon=0., init=0., alpha=0.1, sampleAvg=False, c=None, sga=False, sgaBaseline=False, trueReward=0.):
        self.k = k
        self.stepSize = alpha
        self.sampleAverages = sampleAvg
        self.epsilon = epsilon
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCBParam = c
        self.gradient = sga
        self.gradientBaseline = sgaBaseline
        self.averageReward = 0
        self.trueReward = trueReward
        # real reward for each action
        self.qTrue = []
        # estimation for each action
        self.qEst = np.zeros(self.k)
        # # of chosen times for each action
        self.actionCount = []
        # initialize real rewards with N(0,1) distribution and estimations with desired initial value
        for i in range(0, self.k):
            self.qTrue.append(np.random.randn() + trueReward)
            self.qEst[i] = init
            self.actionCount.append(0)
        self.bestAction = np.argmax(self.qTrue)

    # pick an action for this bandit, explore or exploit?
    def selectAction(self):
        # explore
        if self.epsilon > 0:
            if np.random.binomial(1, self.epsilon) == 1: # When you do not want to be greedy,...
                return np.random.choice(self.indices) # ...return an action chosen at random
        # exploit
        if self.UCBParam is not None:
            UCBEst = self.qEst + self.UCBParam * np.sqrt(np.log(self.time + 1) / (np.asarray(self.actionCount) + 1))
            return np.argmax(UCBEst) # return the index with the max value UCBEst
        if self.gradient:
            expEst = np.exp(self.qEst)
            self.actionProb = expEst / np.sum(expEst)
            return np.random.choice(self.indices, p=self.actionProb) # return a random index basis this probability
        return np.argmax(self.qEst) # else, return the index with the max value. This is the regular, simple, greedy approach

    # take an action, update estimation for this action
    def takeAction(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.qTrue[action]
        self.time += 1
        self.averageReward = ((self.time - 1.0) * self.averageReward + reward)/ self.time
        self.actionCount[action] += 1

        if self.sampleAverages:
            # update estimation using sample averages
            self.qEst[action] += 1.0 / self.actionCount[action] * (reward - self.qEst[action])
        elif self.gradient:
            oneHot = np.zeros(self.k)
            oneHot[action] = 1
            if self.gradientBaseline:
                baseline = self.averageReward
            else:
                baseline = 0
            self.qEst = self.qEst + self.stepSize * (reward - baseline) * (oneHot - self.actionProb)
        else:
            # update estimation with constant step size
            self.qEst[action] += self.stepSize * (reward - self.qEst[action])
        return reward
    
figureIndex = 0

# for figure 2.1
def kArmTestBed(k):
    global figureIndex
    plt.figure(figureIndex)
    figureIndex += 1
    np.random.seed(12345)
    sns.violinplot(data=np.random.randn(2000,k) + np.random.randn(k), color="grey", inner=None)
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")

def epsilonPerf(runs, time, eps, q0):
    banditRuns = []
    banditRuns.extend(Bandit(epsilon=eps, sampleAvg=True, init=q0) for b in range(0, runs))
    # For each bandit, a numpy array with "time" number of elements to hold bestActionCounts and averageRewards
    bestActionCounts = [np.zeros(time, dtype='float') for _ in range(0, runs)]
    averageRewards = [np.zeros(time, dtype='float') for _ in range(0, runs)]
    # For each Bandit class object in the bandits list,     
    for i in range(0, runs):
        for t in range(0, time):
            action = banditRuns[i].selectAction()
            reward = banditRuns[i].takeAction(action)
            averageRewards[i][t] += reward
            if action == banditRuns[i].bestAction:
                bestActionCounts[i][t] += 1
    bestActionCounts = sum(bestActionCounts)/runs
    averageRewards = sum(averageRewards)/runs
    return bestActionCounts, averageRewards

# for figure 2.2
def perfPlot():
    epsilons = [0, 0.1, 0.01]
    runs = 2000
    time = 1000
    global figureIndex
    currentfigIndex = figureIndex
    figureIndex += 1
    nextfigIndex = figureIndex
    for eps in epsilons:
        q0 = 0
        bestActionCounts, averageRewards = epsilonPerf(runs, time, eps, q0)
        plt.figure(currentfigIndex)
        plt.plot(bestActionCounts, label='epsilon = '+str(eps))
        plt.xlabel('Steps')
        plt.ylabel('% optimal action')
        plt.legend()
        plt.figure(nextfigIndex)
        plt.plot(averageRewards, label='epsilon = '+str(eps))
        plt.xlabel('Steps')
        plt.ylabel('average reward')
        plt.legend()


def epsilonPerf2(runs, time, eps, q0, myAlpha):
    banditRuns = []
    banditRuns.extend(Bandit(epsilon=eps, sampleAvg=False, init=q0, alpha=myAlpha) for b in range(0, runs))
    # For each bandit, a numpy array with "time" number of elements to hold bestActionCounts and averageRewards
    bestActionCounts = [np.zeros(time, dtype='float') for _ in range(0, runs)]
    averageRewards = [np.zeros(time, dtype='float') for _ in range(0, runs)]
    # For each Bandit class object in the bandits list,     
    for i in range(0, runs):
        for t in range(0, time):
            action = banditRuns[i].selectAction()
            reward = banditRuns[i].takeAction(action)
            averageRewards[i][t] += reward
            if action == banditRuns[i].bestAction:
                bestActionCounts[i][t] += 1
    bestActionCounts = sum(bestActionCounts)/runs
    averageRewards = sum(averageRewards)/runs
    return bestActionCounts, averageRewards


# for figure 2.3
def perfPlot2():
    runs = 2000
    time = 1000
    global figureIndex
    figureIndex += 1
    plt.figure(figureIndex)
    bestActionCounts1, _ = epsilonPerf2(runs, time, eps=0, q0=5, myAlpha=0.1)
    plt.plot(bestActionCounts1, label='Initial value = '+str(5))
    bestActionCounts2, _ = epsilonPerf2(runs, time, eps=0.1, q0=5, myAlpha=0.1)        
    plt.plot(bestActionCounts2, label='Initial value = '+str(0))        
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()
    
kArmTestBed(10)
perfPlot()
perfPlot2()
