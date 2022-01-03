import torch

from stable_baselines3 import SAC
from functions.simulator import SimulatorCallbacks
from functions.teleoperation import TeleoperationCallbacks
from learning_racer.agent import ControlCallback
from learning_racer.agent.agent import Agent
from learning_racer.exce.LearningRacerError import OptionsValueError
from learning_racer.robot import JetbotEnv, JetRacerEnv
from learning_racer.sac import reward_sim, reward, CustomSAC
from learning_racer.teleoperate import Teleoperator
from learning_racer.vae.vae import VAE
from learning_racer.robot.donkey_sim.donkey_sim_env import factory_creator
from logging import getLogger

logger = getLogger(__name__)

robot_drivers = {'jetbot': JetbotEnv, 'jetracer': JetRacerEnv, 'sim': factory_creator}


def _load_vae(model_path, variants_size, image_channels, device):
    vae = VAE(image_channels=image_channels, z_dim=variants_size)
    try:
        vae.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    except FileNotFoundError:
        logger.error("Specify VAE model path can not find. Please specify correct vae path using -vae option.")
        raise OptionsValueError(
            "Specify VAE model path can not find. Please specify correct vae path using -vae option.")
    vae.to(torch.device(device)).eval()
    return vae


def _init_agent(args, config, train=True):
    torch_device = args.device
    vae = _load_vae(args.vae_path, config.sac_variants_size(), config.sac_image_channel(), torch_device)
    print(args.robot_driver)
    agent = None
    if args.robot_driver in ['jetbot', 'jetracer']:
        teleop = Teleoperator()
        teleop.start_process()
        callbacks = TeleoperationCallbacks(agent, config, teleop)
        env = robot_drivers[args.robot_driver]()
        agent = Agent(env, vae, teleop=teleop, device=torch_device, config=config, train=train, callbacks=callbacks)
    elif args.robot_driver == 'sim':
        driver = robot_drivers[args.robot_driver](args.sim_path, args.sim_host, args.sim_port, args.sim_track)
        env = driver()
        #        env.set_reward_fn(reward_sim)
        callbacks = SimulatorCallbacks(config, env)
        agent = Agent(env, vae, device=torch_device, config=config, train=train, callbacks=callbacks)
    else:
        logger.error("{} is not support robot name.".format(args.robot_driver))
        exit(-1)
    return agent


def command_train(args, config):
    agent = _init_agent(args, config)
    model = CustomSAC(agent, args, config)
    model.lean(callback=agent.callbacks)
    model.save(args.save)


def command_demo(args, config):
    agent = _init_agent(args, config, train=False)
    model = SAC.load(args.model_path)
    obs = agent.reset()
    for step in range(args.time_steps):
        if step % 100 == 0: print("step: ", step)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = agent.step(action)
