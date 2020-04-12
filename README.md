LearningRacer-rl
======

Overview

This software is able to Self learning your AI RC Car 
by Deep reinforcement learning in few minutes.

![demo](content/demo.gif)

And can learning DonkeySim. [See in](#simulator).

## 1. Description

DIY self driving car like JetBot or JetRacer, DonkeyCar are  learning by supervised-learning.
The method need much labeled data that written human. Running behavior characteristic is determined that data.

Deep reinforcement learning (DRL) is can earned running behavior automatically through interaction with environment.
Do not need sample data that is human labelling.

This is using Soft Actor Critic as DRL algorithm. The algorithm is State of The Art of DRL in real environment.
In addition, using Variational Auto Encoder(VAE) as State representation learning. 
VAE can compress environment information, can speed up learning.


* This method devised by Arrafin
    * [Arrafine's Medium blog post](https://towardsdatascience.com/learning-to-drive-smoothly-in-minutes-450a7cdb35f4)
    * [Arrafine's implementsation for Simulator](https://github.com/araffin/learning-to-drive-in-5-minutes)


* About Soft actor critic
    * [Google AI blog Soft Actor-Critic: Deep Reinforcement Learning for Robotics](https://ai.googleblog.com/2019/01/soft-actor-critic-deep-reinforcement.html)

## 2. Demo

This video is 
JetBot is learning running behavior on road in under 30 minutes. Software is running on Jetson Nano.  

[![](https://img.youtube.com/vi/j8rSWvcO-s4/0.jpg)](https://www.youtube.com/watch?v=j8rSWvcO-s4)


## 3. Setup

### 3.1 Requirements

Please install below requirements by manually.

When using JetBot or JetRacer.

* JetBot or JetRacer base image(Recommend latest images)
* tensorflow-gpu=1.14.0
* torch=1.3.0
* torchvision=0.4.2
* OpenCV=4.1.1

When using DonkeySim.

* tensorflow>=1.15.0 (recommend tensorflow-gpu)
* torch=1.4.0
* torchvision=0.5.0
* opencv-python>=4.1.1
* gym_donkey==latest

### 3.2 Install

#### Dependency library install.

Only JetBot and JetRacer.

```
$sudo apt install -y liblapack-dev scipy
```

#### Install racer command.

```shell
$ cd ~/ && git clone https://github.com/masato-ka/airc-rl-agent.git
$ cd airc-rl-agent
$ sudo pip3 install .
```

When complete install please check run command.

```shell
$ racer --version
learning_racer version 1.0.0 .
```

## 4. Usage

### 4.1 JetBot and JetRacer

#### Create VAE Model

1. Collect Environment data as 1k to 10 k images using ```data_collection.ipynb``` or ```data_collection_without_gamepad.ipynb```in ```notebook/utility/jetbot```.
If you use on JetRacer, use```notebook/utility/jetracer/data_collection.ipynb``` . 
2. Learning VAE using ```VAE CNN.ipynb``` on Google Colaboratory.
3. Download vae.torch from host machine and deploy to root directory.

#### Check and Evaluation


Run ```notebooks/util/jetbot_vae_viewer.ipynb``` and Check reconstruction image.
Check that the image is reconstructed at several places on the course.

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

* racer train options

|Name           | description            |Default                |
|:--------------|:-----------------------|:----------------------|
|-config(--config-path)| Specify the file path of config.yml.    | config.yml             |
|-vae(--vae-path)| Specify the file path of the trained VAE model.    | vae.torch             |
|-device(--device)|Specifies whether Pytorch uses CUDA. Set 'cuda' to use. Set 'cpu' when using CPU.| cuda                 |
|-robot(--robot-driver)| Specify the type of car to use. JetBot and JetRacer can be specified.| JetBot              |
|-steps(--time-steps)| Specify the maximum learning step for reinforcement learning. Modify the values ​​according to the size and complexity of the course.| 5000 |
|-save_freq(--save_freq_episode) | 
Specify how many episodes to save the policy model. The policy starts saving after the gradient calculation starts.| 10|
|-s(--save)    | Specify the path and file name to save the model file of the training result.  | model                 |
|-l(--load-model)|Define pre-train model path.|-|

#### Running DEMO

You can running your car without learning. Run below command, The script load vae model and RL model 
and start controll your car.

```shell
$ racer demo -robot jetbot
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


In below command, run the demo 1000 steps with model file name is model.

```shell
$ racer demo -robot jetbot -steps 1000 -model model
```
### <a name="simulator"></a> 4.1 Simulator


#### Download VAE model.

You can get pre-trained VAE model. from [here](https://drive.google.com/open?id=19r1yuwiRGGV-BjzjoCzwX8zmA8ZKFNcC)

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
|-save_freq(--save_freq_episode) |
Specify how many episodes to save the policy model. The policy starts saving after the gradient calculation starts.| 10|
|-host(--sim-host)|Define host IP of DonkeySim host.|127.0.0,1|
|-port(--sim-port)|Define port number of DonkeySim host.|9091|
|-track(--sim-track)|Define track name for DonkeySim.|donkey-generated-trach-v0|
|-s(--save)    | Specify the path and file name to save the model file of the training result.  | model                 |

#### Start Demo

```shell
$ racer demo -robot sim -model <own trained model path> -vae <downloaded vae model path> -steps 1000 -device cpu -host <DonkeySim IP>
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
|-host(--sim-host)|Define host IP of DonkeySim host.|127.0.0,1|
|-port(--sim-port)|Define port number of DonkeySim host.|9091|
|-track(--sim-track)|Define track name for DonkeySim.|donkey-generated-trach-v0|

## 5. Appendix

### 5.1 Configuration

You can configuration hyperparameter using config.yml.

|Section          |Parameter              |Description               |
|:----------------|:----------------------|:-------------------------|
|SAC_SETTING      |LOG_INTERVAL           | [Reference to stable baselines document.](https://stable-baselines.readthedocs.io/en/master/modules/sac.html)             |
|^                |VERBOSE                | ^                        |
|^                |LERNING_RATE           | ^                        |
|^                |ENT_COEF               | ^                        |
|^                |TRAIN_FREQ             | ^                        |
|^                |BATCH_SIZE             | ^                        |
|^                |GRADIENT_STEPS         | ^                        |
|^                |LEARNING_STARTS        | ^                        |
|^                |BUFFER_SIZE            | ^                        |
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

## 7. Contribution

* If you find bug or want to new functions, Please write issue.
* If you fix your self, please fork and send pull request.

## LICENSE

This software license under [MIT](https://github.com/masato-ka/airc-rl-agent/blob/master/LICENCE) licence.

## Author

[masato-ka](https://github.com/masato-ka)
