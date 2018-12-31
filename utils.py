import numpy as np

#
from matplotlib import pyplot as plt
from IPython.display import clear_output


def plot(frame_idx, rewards, losses):
    # clear_output(True)

    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.savefig('./stats.png')
    plt.show()
