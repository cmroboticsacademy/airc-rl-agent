LearningRacer-rl
======

Overview

This software is able to self learning your AI Robocar by Deep reinforcement learning in few minutes.

![demo](content/demo.gif)

You can use to Real Robocar and DonkeySim [See in](#simulator).

## 1. Description

Many DIY self driving car like JetBot or JetRacer, DonkeyCar are  using behavior cloning  by supervised-learning.
The method need much labeled data that is collected by human demonstration.  Human driving techniques is very important in this case.

On the other hands, In this software using deep reinforcement learning (DRL). 
That is can earned running behavior automatically through interaction with environment. Do not need sample data that is human labelling.

In addition this software agent can run on the Jetson Nano. Why can run on Jetson Nano and short learning time? because using integrate of SAC[soft actor critic] and VAE. SAC is a state of the art off-policy reinforcement learning method.
In addition VAE train on cloud server beforehand as CNN layer of SAC.(This method called state representation learning) .


* This method devised by Antonin RAFFIN
    * [Arrafine's Medium blog post](https://towardsdatascience.com/learning-to-drive-smoothly-in-minutes-450a7cdb35f4)
    * [Arrafine's implementsation for Simulator](https://github.com/araffin/learning-to-drive-in-5-minutes)


* Detail of SAC here:
    * [Google AI blog Soft Actor-Critic: Deep Reinforcement Learning for Robotics](https://ai.googleblog.com/2019/01/soft-actor-critic-deep-reinforcement.html)

## 2. Demo

This demo video showed that JetBot can earned policy of running road under 30 minutes. Only using Jetson Nano. 

[![](https://img.youtube.com/vi/j8rSWvcO-s4/0.jpg)](https://www.youtube.com/watch?v=j8rSWvcO-s4)


## 3. Setup

### 3.1 Requirements

* Jetbot or JetRacer
* JetPack>=4.2
* Python=>3.6
* pip>=19.3.1
* pytorch>=1.8.0

* Windows, macOS or Ubuntu (DonkeySim only)
* x86-64 arch
* Python>=3.6
* pip>=19.3.1
* DonkeySIM
* Optional CUDA10.1(Windows and using GPU.)
* pytorch>=1.8.0

### 3.2 Install

* JetBot

Set up JetBot using the following SDCard image.
[https://jetbot.org/v0.4.3/software_setup/sd_card.html]

Checking your JetBot Environment. Please write down JETBOT_VERSION and L4T_VERSION.

```
#JETBOT_VERSION
$ sudo docker images jetbot/jetbot | grep jupyter | cut -f 8 -d ' ' | cut -f 2 -d '-'

#L4T_VERSION
$ sudo docker images jetbot/jetbot | grep jupyter | cut -f 8 -d ' ' | cut -f 3 -d '-'

```

And Setup LearningRacer for Docker container image.

```
$ cd ~/ && git clone https://github.com/masato-ka/airc-rl-agent.git
$ cd airc-rl-agent/docker/jetbot && sh build.sh
$ sh enable.sh /home/jetbot

# disable jetbot/jetbot container. Tag name modify for your system by JETBOT_VERSION and L4T_VERSION.
$ sudo docker update --restart=no jetbot_jupyter
$ sudo restart
```

JetBot images(JetPack>=4.4) are using docker container . Therefore, build application on docker container . allocate
maximum memory to the container.

You are able to use ```racer``` command inside docker container. Access to Jupyter Notebook on the
container[http://<jetbot-ip>:8888/] and launch terminal(File->new->terminal ).

You need train original VAE model. Because torch version problem. Coud you cahange
to ```torch.save(vae.state_dict(), 'vae.torch', _use_new_zipfile_serialization=True)``` in VAE_CNN.ipynb training cell.

* JetRacer.

Firstly setup your jetracer software to JetPack 4.5.1 following
this [link](https://github.com/NVIDIA-AI-IOT/jetracer/blob/master/docs/software_setup.md). Then run below command on
your jetracer terminal.

```
$ cd ~/ && git clone https://github.com/masato-ka/airc-rl-agent.git
$ cd airc-rl-agent
$ sh install_jetpack.sh
```

Some time pytorch can not recognize your GPU by CUDA Driver problem. In this situation, you need to install pytorch
following this [link](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048). Detail
see
in [this](https://forums.developer.nvidia.com/t/my-jetson-nano-board-returns-false-to-torch-cuda-is-available-in-local-directory/182498)

* Other platform(DonkeySIM only).

```
$ cd ~/ && git clone https://github.com/masato-ka/airc-rl-agent.git
$ cd airc-rl-agent
$ sudo pip3 install .\[choose platform\]
```

* You can choose platform from here
    * windows
    * windows-gpu
    * osx
    * ubuntu


When complete install please check run command.

```shell
$ racer --version
learning_racer version 1.5.0 .
```

## 4. Usage

### 4.1 JetBot and JetRacer

#### Create VAE Model

1. Collect 1k to 10 k images from your car camera using ```data_collection.ipynb```
   or ```data_collection_without_gamepad.ipynb```in ```notebook/utility/jetbot```. If you use on JetRacer,
   use```notebook/utility/jetracer/data_collection.ipynb``` .
2. Learning VAE using ```VAE CNN.ipynb``` on Google Colaboratory.
3. Download vae.torch from host machine and deploy to root directory.

When your robot is Jetbot, Coud you modify VAE_CNN.ipynb.

* final line trainng cell, Please change to True.
  '''
  torch.save(vae.state_dict(), 'vae.torch', _use_new_zipfile_serialization=True)
  '''

#### Check and Evaluation

**A.Offline check**

When you run VAE_CNN.ipynb, you can check projection of latent spaces on TensorBoard Projection Tab. This latent spaces
are labeled by K-means. If similar images stick together, it indicate to that good latent spaces.

![tensorboard-projection](content/vae/tensorboard-projection.png)

**B.Online check**

Run ```notebooks/util/jetbot_vae_viewer.ipynb``` and Check reconstruction image. Check that the image is reconstructed
at several places on the course.

If you use on JetRacer, Using ```jetracer_vae_viewer.ipynb``` .

* Left is an actual image. Right is reconstruction image.
* Color bar is represented latent variable of VAE(z=32 dim).

![vae](content/vae/vae.gif)


#### Start learning

1. Run user_interface.ipynb (needs gamepad).
If you not have gamepad, use ```user_interface_without_gamepad.ipynb```
2. Run train.py

```shell
$ racer train -robot jetbot
# If you use on JetRacer, "-robot jetracer". default is jetbot.
```

After few minutes, the AI car starts running. Please push STOP button immediately before the course out. 
Then, after `` `RESET``` is displayed at the prompt, press the START button. Repeat this.

![learning](content/learning.gif)

When you use without_gamepad, you can check status using Validation box.

|Can run                          | Waiting learning                       |
|:-------------------------------:|:--------------------------------------:|
|![can_run](content/status_ok.png)|![waiting_learn](content/status_ng.png) |

* racer train command options

|Name           | description            |Default                |
|:--------------|:-----------------------|:----------------------|
|-config(--config-path)| Specify the file path of config.yml.    | config.yml             |
|-vae(--vae-path)| Specify the file path of the trained VAE model.    | vae.torch             |
|-device(--device)|Specifies whether Pytorch uses CUDA. Set 'cuda' to use. Set 'cpu' when using CPU.| cuda                 |
|-robot(--robot-driver)| Specify the type of car to use. choose from jetbot, jetracer, jetbot-auto, jetracer-auto and sim.| JetBot              |
|-steps(--time-steps)| Specify the maximum learning step for reinforcement learning. Modify the values ​​according to the size and complexity of the course.| 5000 |
|-save_freq(--save_freq_episode) |
Specify how many episodes to save the policy model. The policy starts saving after the gradient calculation starts.| 10|
|-s(--save)    | Specify the path and file name to save the model file of the training result.  | model                 |
|-l(--load-model)|Define pre-train model path.|-|

In -robot option, If you choose jetracer-auto or jetbot-auto, Auto train mode start. When this mode, Robot stop without
human controll and pullback position where start learning.

#### Running DEMO

When only inference, run below command, The script load VAE model and RL model and start running your car.

```shell
$ racer demo -robot jetbot
``` 

* racer demo command options

|Name           | description            |Default                |
|:--------------|:-----------------------|:----------------------|
|-config(--config-path)| Specify the file path of config.yml.    | config.yml             |
|-vae(--vae-path)| Specify the file path of the trained VAE model.    | vae.torch             |
|-model(--model-path|Specify the file to load the trained reinforcement learning model.|model|
|-device(--device)|Specifies whether Pytorch uses CUDA. Set 'cuda' to use. Set 'cpu' when using CPU.| cuda                 |
|-robot(--robot-driver)| Specify the type of car to use. JetBot and JetRacer can be specified.| JetBot              |
|-steps(--time-steps)| Specify the maximum step for demo. Modify the values ​​according to the size and complexity of the course.| 5000 |
|-tblog(--tb-log)|Define logging directory name, If not set, Do not logging.|None|

In below command, run the demo 1000 steps with model file name is model.

```shell
$ racer demo -robot jetbot -steps 1000 -model model
```

### <a name="simulator"></a> 4.1 Simulator

#### Download VAE model.

You can get pre-trained VAE model. from [here](https://drive.google.com/open?id=19r1yuwiRGGV-BjzjoCzwX8zmA8ZKFNcC)

```shell
$wget "https://drive.google.com/uc?export=download&id=19r1yuwiRGGV-BjzjoCzwX8zmA8ZKFNcC" -O vae.torch
```

#### Start learning

```shell
$ racer train -robot sim -vae <downloaded vae model path> -device cpu -host <DonkeySim IP>
```

* racer train options

|Name           | description            |Default                |
|:--------------|:-----------------------|:----------------------|
|-config(--config-path)| Specify the file path of config.yml.    | config.yml             |
|-vae(--vae-path)| Specify the file path of the trained VAE model.    | vae.torch             |
|-device(--device)|Specifies whether Pytorch uses CUDA. Set 'cuda' to use. Set 'cpu' when using CPU.| cuda                 |
|-robot(--robot-driver)| Specify the type of car to use. JetBot and JetRacer can be specified.| JetBot              |
|-steps(--time-steps)| Specify the maximum learning step for reinforcement learning. Modify the values ​​according to the size and complexity of the course.| 5000 |
|-save_freq(--save_freq_episode) |Specify how many steps to save the policy model. The policy starts saving after the gradient calculation starts.| 10|
|-save_path(--save-model-path)| Specify the path for saved model file.|model_log|
|-s(--save)    | Specify the path and file name to save the model file of the training result.  | model                 |
|-l(--load-model)|Define pre-train model path.|-|

#### Start Demo

```shell
$ racer demo -robot sim -model <own trained model path> -vae <downloaded vae model path> -steps 1000 -device cpu -host <DonkeySim IP> -user <your own name>
```

* racer demo options

|Name           | description            |Default                |
|:--------------|:-----------------------|:----------------------|
|-config(--config-path)| Specify the file path of config.yml.    | config.yml             |
|-vae(--vae-path)| Specify the file path of the trained VAE model.    | vae.torch             |
|-model(--model-path|Specify the file to load the trained reinforcement learning model.|model|
|-device(--device)|Specifies whether Pytorch uses CUDA. Set 'cuda' to use. Set 'cpu' when using CPU.| cuda                 |
|-robot(--robot-driver)| Specify the type of car to use. JetBot and JetRacer can be specified.| JetBot              |
|-steps(--time-steps)| Specify the maximum step for demo. Modify the values ​​according to the size and complexity of the course.| 5000 |
|-user(--sim-user)  |Define user name for own car that showed DonkeySim |anonymous|
|-car(--sim-car)    | Define car model type for own car that showed DonkeySim|Donkey|

## 5. Appendix

### 5.1 Configuration

You can configuration to some hyper parameter using config.yml.

|Section          |Parameter              |Description               |
|:----------------|:----------------------|:-------------------------|
|SAC_SETTING      |LOG_INTERVAL           | [Reference to stable baselines document.](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)             |
|^                |VERBOSE                | ^                        |
|^                |LERNING_RATE           | ^                        |
|^                |ENT_COEF               | ^                        |
|^                |TRAIN_FREQ             | ^                        |
|^                |BATCH_SIZE             | ^                        |
|^                |GRADIENT_STEPS         | ^                        |
|^                |LEARNING_STARTS        | ^                        |
|^                |BUFFER_SIZE            | ^                        |
|^                |GAMMA                  | ^                        |
|^                |TAU                    | ^                        |
|^                |USER_SDE               | ^                        |
|^                |USER_SDE_AT_WARMUP     | ^                        |
|^                |SDE_SAMPLE_FREQ        | ^                        |
|^                |VARIANTS_SIZE          | Define size of VAE latent|
|^                |IMAGE_CHANNELS         | Number of image channel. |
|REWARD_SETTING   |REWARD_CRASH           | Define reward when crash.|
|^                |CRASH_REWARD_WEIGHT    | Weight of crash reward.   |
|^                |THROTTLE_REWARD_WEIGHT | Weight of reward for speed. |
|AGENT_SETTING    |N_COMMAND_HISTORY      | Number of length command history as observation.|
|^                |MIN_STEERING           | min value of agent steering.|
|^                |MAX_STEERING           | max value of agent steering.|
|^                |MIN_THROTTLE           | min value of agent throttle.|
|^                |MAX_THROTTLE           | max value of agent throttle.|
|^                |MAX_STEERING_DIFF      | max value of steering diff each steps.| 
|JETRACER_SETTING |STEERING_CHANNEL       | Steering PWM pin number.|
|^                |THROTTLE_CHANNEL       | Throttle PWM pin number.|
|^                |STEERING_GAIN          | value of steering gain for NvidiaCar.|
|^                |STEERING_OFFSET        | value of steering offset for NvidiaCar.|
|^                |THROTTLE_GAIN          | value of throttle gain for NvidiaCar.|
|^                |THROTTLE_OFFSET        | value of throttle offset for NvidiaCar.| 



## 6. Release note

* 2020/03/08 Alpha release
    * First release.
    
* 2020/03/16 Alpha-0.0.1 release
    * Fix import error at jetbot_data_collection.ipynb.

* 2020/03/23 Beta release
    * VAE Viewer can see latent space.
    * Avoid stable_baseline source code change at install.
    * train.py and demo.py merged to racer.py.
    * Available without a game controller.
    * Fix for can not copy dataset from google drive in CNN_VAE.ipynb

* 2020/03/23 Beta-0.0.1 release
    * Fix VAE_CNN.ipynb (bug #18).

* 2020/04/26 v1.0.0 release
    * Improvement install function.
    * Can use DonkeySIM.
    * YAML base configuration.
    * Can use pre-trained model for SAC.
    * Periodical saved model in each specific episode.

* 2020/06/30 v1.0.5 release
    * BugFix
        * \#20 Recording twice action in a step
        * \#22 Error occurs when run demo subcommand in real car.
    * Jetson nano install script improvement.
    
* 2020/10/11 v1.5.0 release
    * BugFix
        * \#25 Change interface for latest gym_donkey
    * Migration to stable_baseline3(All Pytorch implementation)
    * Improvement notebook.
        * user_interface.ipynb change UI.
        * VAE_CNN.ipynb a little faster training.
    * You can use TensorBoard for monitoring training.

* 2021/01/09 v1.5.1 release
    * BugFix
        * \#32 Dose not working on Simulator environment.
        * README.md update.

* 2021/04/11 v1.5.2 release
    * Corresponding to stable_baseline3 1.0

* 2021/12/26 v1.6.0 release
    * BugFix
        * \#33 Fix Can not stop over episode in simulator.
        * \#40 Fix vae_viewer.ipynb is failed visualize reconstruction image color.
        * \#42 Fix Can not load trained model.
        * \#43 Fix VAE model problems.
    * Improvement Notebook
        * Visualize latent space with TensorBoar Projection(VAE_CNN.ipynb)
    * Improvement Function.
        * Docker installation for JetBot.

* 2022/03/27 v1.7.0 release
    * BugFix
        * \#46 In simurator crush reward fix.
    * Improvement Function.
        * Release auto stop function without detail document.
    * Other
        * VAE model is changed. VAE models are not backward compatible.

* 2022/07/03 v1.7.1 relase
    * BugFix
        * \#47 The learning_racer.vae decoder method is outputting the distributed image incorrectly.
        * \#38 CNN_VAE.ipynb fix.
        * vae_viewer.ipynb fix.
        * Can not start command for internal errors.
    * Inprovement Function.
        * Efficient hyperparameter in config.yml

## 7. Contribution

* If you find bug or want to new functions, please write issue.
* If you fix your self, please fork and send pull request.

## LICENSE

This software license under [MIT](https://github.com/masato-ka/airc-rl-agent/blob/master/LICENCE) licence.

## Author

[masato-ka](https://github.com/masato-ka)
