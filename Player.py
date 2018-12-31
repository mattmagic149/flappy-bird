import time
from threading import Thread

#
from Game import Game
from Learner import Learner

#
from albumentations.pytorch.functional import img_to_tensor

#
from logging import getLogger, INFO
logger = getLogger('rl-logger')
logger.setLevel(INFO)


class Player(Thread):

    def __init__(self, thread_id, learner, game_options, fps=10, save=False):
        Thread.__init__(self)

        #
        self.thread_id = thread_id

        #
        self.learner = learner

        #
        self.game = Game(game_options, fps=fps, save=save)

    def run(self):

        self.game.init_game()
        state = self.game.start_game()
        episode_reward = 0

        while True:
            logger.info("Next frame.")
            epsilon, learning = self.learner.epsilon

            # all learning iterations done
            if not learning:
                break

            # get best or random action with probability epsilon
            action = self.learner.act(state, epsilon)

            next_state, reward, done = self.game.move(action)
            self.learner.remember(
                img_to_tensor(state),
                action,
                reward,
                img_to_tensor(next_state),
                done
            )

            state = next_state
            episode_reward += reward

            if done:
                logger.warning("Game over ({} fps).".format(len(self.game.frames) / (time.time() - self.game.started)))
                self.learner.update()
                self.learner.add_episode_reward(episode_reward)
                episode_reward = 0

                # reset
                self.game.reset()
                state = self.game.start_game()


def main():

    #
    import selenium.webdriver as webdriver

    thread_id = 1
    input_shape = (3, 128, 128)
    action_space = 2
    capacity = 100000

    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('--mute-audio')
    options.add_argument('window-size=400x600')

    learner = Learner(
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
    )

    logger.warning('Starting threads.')
    threads = []

    # start threads
    learner.start()
    threads.append(learner)
    for i in range(2):
        player = Player(
            2,
            learner,
            options,
            save=False
        )
        player.start()
        threads.append(player)

    logger.warning('All threads started.')

    # Wait for all threads to complete
    for t in threads:
        t.join()
    print("Exiting Main Thread")


if __name__ == "__main__":
    main()

