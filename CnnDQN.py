import random

#
import torch
import torch.nn as nn
import torch.autograd as autograd

#
from albumentations.pytorch.functional import img_to_tensor

#
import numpy as np


class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions, use_cuda=False):
        super(CnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        self.Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if use_cuda else \
            autograd.Variable(*args, **kwargs)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state_ = self.Variable(img_to_tensor(state).unsqueeze(0))
            q_value = self.forward(state_)
            return q_value.max(1)[1].data[0]
        else:
            return random.randrange(self.num_actions)


def main():

    input_shape = (3, 128, 128)
    num_actions = 2
    model = CnnDQN(input_shape, num_actions)
    x = np.ones((128, 128, 3))

    #
    y1 = model.act(x, 0.)
    print('y1 =', np.array(y1))

    #
    y2 = model.act(x, 1.)
    print('y2 =', y2)


if __name__ == "__main__":
    main()
