import time

import numpy as np
from gym import Env

from learning_racer.config import ConfigReader
from learning_racer.agent import BaseWrappedEnv
from learning_racer.teleoperate import Teleoperator

from logging import getLogger

from learning_racer.vae import VAE

logger = getLogger(__name__)


def real_world_reward(action, done, min_throttle, max_throttle,
                      crash_reward, crash_reward_weight, throttle_reward_weight):
    """

    :param action: tuple of throttle and steering
    :param done: boolean
    :param min_throttle: float
    :param max_throttle: float
    :param crash_reward: float
    :param crash_reward_weight: float
    :param throttle_reward_weight: float
    :return: float and boolean
    """
    if done:
        norm_throttle = (action[1] - min_throttle) / (
                max_throttle - min_throttle)
        return crash_reward - (crash_reward_weight * norm_throttle), done
    throttle_reward = throttle_reward_weight * (action[1] / max_throttle)
    return 1 + throttle_reward, done


class TeleoperationEnv(BaseWrappedEnv):

    def __init__(self, env: Env, vae: VAE, config: ConfigReader, teleoperator: Teleoperator):
        super(TeleoperationEnv, self).__init__(env, vae, config)
        self.teleoperator = teleoperator
        self.teleoperator.start_process()

    def on_training_end(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        if self.teleoperator is not None:
            self.teleoperator.send_status(True)
            message = True
            while self.teleoperator.status:
                if message:
                    logger.info("Press START.")
                message = False
                time.sleep(0.1)

    def on_pre_step_callback(self, action):
        return action

    def on_post_step_callback(self, action, t_img, reward, done, info, z, train):
        # Override Done event.
        done = self._done_override(action, t_img, reward, done, info, z, train)
        # Override Reward value.
        reward, done = real_world_reward(action, done, self.config.agent_min_throttle(),
                                         self.config.agent_max_throttle(),
                                         self.config.reward_reward_crash(), self.config.reward_crash_reward_weight(),
                                         self.config.reward_throttle_reward_weight())

        return action, t_img, reward, done, info, z

    def on_pre_reset(self):
        return None

    def on_post_reset(self, observe):
        return observe

    def _done_override(self, action, observe, reward, done, info, z, train):
        if self.teleoperator is not None:
            done = self.teleoperator.status
            if done and train:
                self.env.step(np.array([0., 0.]))
                self.teleoperator.send_status(False)
        return done
