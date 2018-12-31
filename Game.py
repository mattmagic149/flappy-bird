import os
import io

#
import time
from datetime import datetime, timedelta

#
import numpy as np

#
import selenium.webdriver as webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

#
import cv2
from PIL import Image

#
from logging import getLogger, INFO
logger = getLogger('rl-logger')
logger.setLevel(INFO)


class Game:

    def __init__(self,
                 options,
                 fps=10,
                 save=False,
                 data_dir='data/games',
                 image_size=(128, 128),
                 positive_reward=1,
                 negative_reward=-1
                 ):

        # open browser
        print(os.getcwd())
        self.driver = webdriver.Chrome(
            options=options,
            executable_path='./chromedriver'
        )
        self.driver.set_window_size(320, 720)
        self.driver.get('https://flappybird.online/')

        #
        self.board = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "flyarea"))
        )

        self.score_elem = self.board.find_element_by_xpath('//div[@id="bigscore"]')
        self.player_elem = self.board.find_element_by_xpath('//div[@id="player"]')
        self.scoreboard_elem = self.board.find_element_by_xpath('//div[@id="scoreboard"]')

        #
        self.id = hex(int(time.time()))
        self.last_frame_captured = datetime.utcnow()
        self.fps = fps
        self.spf = 1. / fps
        self.image_size = image_size

        #
        self.frames = []
        self.score = None
        self.previous_score = None

        #
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

        #
        self.save = save
        self.data_dir = data_dir

        # game started at
        self.started = time.time()

        # create game directory
        if self.save:
            os.mkdir(os.path.join(self.data_dir, self.id))

    def init_game(self):
        while True:
            # break on init image
            if self.is_lobby():
                logger.info('init_game: while is_lobby')
                break

            # break on init image
            if self.is_game_over():
                logger.info('init_game: while is_game_over ({})'.format(self.player_state()))
                self.press_space()
                time.sleep(1)

    def reset(self):

        #
        logger.info("resetting game")
        self.id = hex(int(time.time()))
        self.last_frame_captured = datetime.utcnow()

        #
        self.frames = []
        self.score = None
        self.previous_score = None

        #
        self.init_game()

    def start_game(self):
        while self.is_game_over():
            self.init_game()

        while self.is_lobby():
            self.press_space()

        #
        self.started = time.time()
        frame = self.grab_frame()
        self.frames.append(frame)
        return frame

    def is_game_over(self):
        return self.player_state() == 'paused'

    def is_lobby(self):
        return self.current_score() is None and self.player_state() == 'running'

    def is_started(self):
        return self.current_score() is not None and self.player_state() == 'running'

    def parse_style(self, style):
        def split(x):
            return x.split(':')

        def filter_empty(x):
            return x[0] != ''

        def strip(x):
            if len(x) > 1:
                return x[0].strip(), x[1].strip()
            return x

        styles = style.split(';')
        key_values = list(map(strip, filter(filter_empty, map(split, styles))))
        return dict(key_values)

    def player_state(self):
        player_dict = self.parse_style(self.player_elem.get_attribute('style'))
        return player_dict['animation-play-state']

    def wait_next_frame(self):
        next_frame_time = self.last_frame_captured + timedelta(seconds=self.spf)
        diff = next_frame_time - datetime.utcnow()
        if diff.total_seconds() > 0:
            time.sleep(diff.total_seconds())

    def grab_frame(self):

        # wait for next frame to grab
        self.wait_next_frame()

        # grab frame as bytes and convert them to image
        frame = Image.open(io.BytesIO(self.board.screenshot_as_png))
        frame = frame.resize(self.image_size)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)

        self.last_frame_captured = datetime.utcnow()

        if self.save:
            cv2.imwrite(os.path.join(
                self.data_dir,
                str(self.id),
                '{}.png'.format(str(len(self.frames)))
            ), frame)

        self.frames.append(frame)
        return frame

    def press_space(self):
        action_chains = ActionChains(self.driver)
        action_chains.send_keys(Keys.SPACE)
        action_chains.perform()

    def move(self, move):
        if move == 1:
            # press button
            self.press_space()
        elif move == 0:
            # do nothing
            pass

        next_state = self.grab_frame()
        reward = self.reward()
        done = self.is_game_over()
        return next_state, reward, done

    def reward(self):
        if self.is_game_over():
            return self.negative_reward

        score = self.current_score()
        self.previous_score = self.score
        self.score = score

        if self.previous_score is not None and score is not None and self.previous_score < score:
            return self.positive_reward

        return 0.1 # len(self.frames) / 100.

    def current_score(self):
        try:
            return int(self.score_elem.find_element_by_xpath('./img').get_attribute('alt'))
        except Exception as e:
            return None


def main():
    options = webdriver.ChromeOptions()
    # options.add_argument('headless')
    options.add_argument('--mute-audio')
    options.add_argument('window-size=400x600')

    # set up game
    fps = 10
    game = Game(options, fps=fps, save=True)

    game.init_game()
    game.start_game()


if __name__ == "__main__":
    main()
