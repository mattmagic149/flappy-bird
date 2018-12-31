#
from collections import deque

#
from threading import Lock, Condition
from threading import Thread

#
from CnnDQN import CnnDQN
from utils import plot

#
import torch
import torch.optim as optim
import torch.autograd as autograd

#
import math
import numpy as np
import random

#
from logging import getLogger, INFO
logger = getLogger('rl-logger')
logger.setLevel(INFO)


class Learner(Thread):

    def __init__(self,
                 thread_id,
                 input_shape,
                 action_space,
                 capacity,
                 use_cuda=False,
                 num_frames=1400000,
                 batch_size=32,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_final=0.01,
                 epsilon_decay=30000,
                 network_update_rate=30
                 ):
        Thread.__init__(self)

        #
        self.thread_id = thread_id

        # learner parameters
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.gamma = gamma
        self.e_start = epsilon_start
        self.e_final = epsilon_final
        self.e_decay = epsilon_decay
        self.e_diff = self.e_start - self.e_final

        # model
        self.model = CnnDQN(input_shape, action_space)

        if use_cuda:
            self.model = self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001)

        # retrain network after n episodes
        self.network_update_rate = network_update_rate

        #
        self._frame_count = -1
        self.episodes_played = 0

        # locks
        self._frame_lock = Lock()
        self._learning_cond = Condition()
        self._episode_payed_lock = Lock()

        # stats
        self.losses = []
        self.rewards = []

        # memory
        self.memory = deque(maxlen=capacity)

        #
        self.Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if use_cuda else \
            autograd.Variable(*args, **kwargs)

    @property
    def is_learning(self):
        return self._frame_count < self.num_frames

    @property
    def epsilon(self):
        with self._frame_lock:
            self._frame_count += 1
            scale = math.exp(-1. * self._frame_count / self.e_decay)
            return self.e_final + self.e_diff * scale, self.is_learning

    def remember(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.memory.append((state, action, reward, next_state, done))

    def memory_sample(self):
        state, action, reward, next_state, done = zip(*random.sample(
            self.memory,
            self.batch_size
        ))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def compute_td_loss(self):

        if self.batch_size > len(self.memory):
            return None

        state, action, reward, next_state, done = self.memory_sample()

        state = self.Variable(torch.FloatTensor(state))
        with torch.no_grad():
            next_state = self.Variable(torch.FloatTensor(next_state))

        action = self.Variable(torch.LongTensor(action))
        reward = self.Variable(torch.FloatTensor(reward))
        done = self.Variable(torch.FloatTensor(done))

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - self.Variable(expected_q_value.data)).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.data)

    def add_episode_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        with self._episode_payed_lock:
            self.episodes_played += 1
            played = self.episodes_played

        if played % self.network_update_rate == 0:
            logger.warning('notify all.')
            with self._learning_cond:
                self._learning_cond.notify_all()

    def act(self, state, epsilon):
        return self.model.act(state, epsilon)

    def run(self):
        while self.is_learning:
            with self._learning_cond:
                self._learning_cond.wait()

                # update value network
                self.compute_td_loss()

                #
                torch.save(self.model.state_dict(), 'weights.hdf5')

                # update plots
                logger.warning('Plot reward & loss.')
                plot(self._frame_count, self.rewards, self.losses)
