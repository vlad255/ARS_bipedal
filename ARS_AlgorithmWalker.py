#MidTerm_Task_Move37

import numpy as np
import gym
from gym import wrappers

#Parameters

ENV_NAME = "BipedalWalker-v2"
MONITOR_DIR = "Videos_of_training"
N_STEPS = 10000                        #Number of trainings we will run
MAX_EPISODE_LEN = 20000                #Max. steps of one episode
NUM_DELTAS = 20
NUM_BEST_DELTAS = 20
RECORD_EVERY = 250
assert NUM_BEST_DELTAS <= NUM_DELTAS   # NUM_DELTAS MUST BE HIGHER OF EQUALLY THEN NUM_BEST_DELTAS

ALPHA = 0.02                           #Learning rate
NOISE = 0.05


class Normalizer():
    #Normalizes inputs
    def __init__(self, n_inputs):
        #Here we create empty arrays the size of our inputs
        self.n = np.zeros(n_inputs)
        self.mean = np.zeros(n_inputs)
        self.mean_diff = np.zeros(n_inputs)
        self.var = np.zeros(n_inputs)

    def observe(self, x):
        #From our inputs we gate array average "mean" and calculate the variance
        self.n += 1
        last_mean = self.mean.copy()
        self.mean += ( x - last_mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-4)  # clip so we don't divede by zero

    def normalize(self, state):
        # normalize inputs 
        o_mean = self.mean
        o_std = np.sqrt(self.var)
        return (state - o_mean) / o_std

class Augmented_Random_Search():
    # class for ARS algorithm
    def __init__(self, n_inputs, n_outputs):
        #input state , output actions
        self.n_inputs = n_inputs #length of state
        self.n_outputs = n_outputs #length of an action
        self.weights = np.zeros((n_outputs,n_inputs))
        self.record_video = False
        self.should_record = lambda i: self.record_video

    def generate_deltas(self, N):
        # we get lost of N deltas 
        return [np.random.randn(*self.weights.shape) for _ in range(N)]

    def policy(self, state, delta = None, direction = None):
        # We give it a state and  policy return action to take in that state
        if direction:
            assert direction is "+" or direction is "-"
        if delta is not None:
            if direction == "+":
                return (self.weights + NOISE*delta).dot(state)
            elif direction == "-":
                return (self.weights - NOISE*delta).dot(state)
        else:
            return self.weights.dot(state)

        return None

    def run_episode(self, env, normalizer, delta = None, direction = None, render = False):
        #we simulate an episode and we get total reward in that episode
        total_reward = 0
        state = env.reset()
        for _ in range(MAX_EPISODE_LEN):
            if render:
                env.render()
            normalizer.observe(state)
            state = normalizer.normalize(state)
            action = self.policy(state, delta = delta, direction = direction)
            state, reward, done, _ = env.step(action)
            reward = max(min(reward, 1), -1)
            total_reward += reward
            if done:
                break
        env.env.close()
        return total_reward
        
    def weights_update(self, rollouts, sd_reward):
        # weights update with rollouts and standard deviation
        step = np.zeros(self.weights.shape)
        for r_pos, r_neg, delta in rollouts:
            step += (r_pos - r_neg) * delta
        self.weights += ALPHA / (NUM_BEST_DELTAS * sd_reward)*step

    def set_weights(self, weights):
        assert weights.shape == self.weights.shape
        self.weights = weights

    def train(self, env, normalizer,RECORD_EVERY):
        #Training for our agent using ARS algorithm
        MAX_EPISODE_LEN = env.spec.timestep_limit or MAX_EPISODE_LEN
        #For loop to train for number of times we dicided in parametars
        for step in range(N_STEPS):
            deltas = self.generate_deltas(NUM_DELTAS)
            pos_delta_rewards = [0] * NUM_DELTAS
            neg_delta_rewards = [0] * NUM_DELTAS
            for i in range(NUM_DELTAS):
                pos_delta_rewards[i] = self.run_episode(env, normalizer, deltas[i], "+")
                neg_delta_rewards[i] = self.run_episode(env, normalizer, deltas[i], "-")
            delta_rewards = np.array(pos_delta_rewards + neg_delta_rewards)
            sd_reward = delta_rewards.std() # calculate standard deviation of delta rewards
            rollouts = [(pos_delta_rewards[i], neg_delta_rewards[i], deltas[i]) for i in range(NUM_DELTAS)]
            rollouts.sort(key = lambda x : max(x[0:2]),reverse = True)
            rollouts = rollouts[:NUM_BEST_DELTAS]

            self.weights_update(rollouts, sd_reward) #weights update

            if step % RECORD_EVERY == 0:
                self.record_video = True

            reward = self.run_episode(env, normalizer)
       
            self.record_video = False

            print("Step: #{} Reward: {}".format(step, reward))

#MAIN
            
env = gym.make(ENV_NAME) #create environment for our game

state_len = env.observation_space.shape[0]  #lenght of state
action_len = env.action_space.shape[0]      #length of action

normalizer = Normalizer(state_len)          #state normalizer
ars_agent = Augmented_Random_Search(state_len, action_len) #create our agent

env = gym.wrappers.Monitor(env, MONITOR_DIR, video_callable=ars_agent.should_record, force=False)

ars_agent.train(env, normalizer, RECORD_EVERY) # training for our agent 
