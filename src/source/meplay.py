from baseline_racer import BaselineRacer
import airsimneurips as airsim
import time
from model import ActorCritic
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
#import gym, gym.spaces, gym.utils, gym.utils.seeding
import math
from tensorboardX import SummaryWriter # using log kinda of thing? , though need to check for ctrl+c early crash situtations dont save yet.
import os # for saving model file
from metest import compute_reward
from metest import isDone
from metest import returnGateLocationsNumpy


### DETECT device
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print('Device:', device)

baseline_racer = BaselineRacer(drone_name="drone_1",viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0])

#creating of input something
number_of_inputs = 21 # for now its beta phase , linear velocity of our drone and enemy drone, position of our drone and enemy drone 3+3+3+3=12 enemy 6, our 15 now
number_of_actions = 3 # it was 7 but x,y,z makes more sense than x+,x- etc. hovering action is not neeeded for now?
# also Timing could be a output that later on but now drone will move 1second fixed or 0.5 second or 1.5 second 
model = ActorCritic(number_of_inputs,number_of_actions,HIDDEN_SIZE).to(device)
model.load_state_dict(torch.load("args.name_best_+8.383_27000.dat"))

pts = returnGateLocationsNumpy(baseline_racer)
state = np.ones(number_of_inputs)
baseline_racer.load_level("Soccer_Field_Easy")
baseline_racer.initialize_drone()
baseline_racer.start_odometry_callback_thread()
baseline_racer.start_image_callback_thread()

timeout = 500

baseline_racer.start_race(1)

while not done and timeout>= 0:
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        dist, _ = model(state)
        action = torch.clamp(dist.sample(), -1.0, 1.0)
        """action = torch.clamp(dist.mean(),-1.0,1.0) if deterministic \
            else torch.clamp(dist.sample(),-1.0,1.0)"""
        if(deterministic == True):
            action = dist.mean.detach().cpu().numpy()[0]
            actx = action
            baseline_racer.airsim_client.moveByVelocityAsync(quad_vel.x_val+ actx[0], quad_vel.y_val+actx[1], quad_vel.z_val+actx[2],0.1).join()
        else:
            action = torch.clamp(dist.sample(),-1.0,1.0)
            actx = action.data.cpu().numpy()
            baseline_racer.airsim_client.moveByVelocityAsync(quad_vel.x_val+ actx[0][0], quad_vel.y_val+actx[0][1], quad_vel.z_val+actx[0][2],0.1).join()

        pose_object_temp = baseline_racer.current_position
        pose_object_temp_numpy = np.array([pose_object_temp.x_val,pose_object_temp.y_val,pose_object_temp.z_val])

        linear_vel_object_temp = baseline_racer.current_linear_velocity
        angular_vel_object_temp = baseline_racer.current_angular_velocity
        linear_acc_object_temp = baseline_racer.current_linear_acceleration
        angular_acc_object_temp = baseline_racer.current_angular_acceleration
        enemy = baseline_racer.current_enemy

        next_state = np.array([pose_object_temp.x_val,pose_object_temp.y_val,pose_object_temp.z_val,linear_vel_object_temp.x_val,
        linear_vel_object_temp.y_val,linear_vel_object_temp.z_val,angular_vel_object_temp.x_val,angular_vel_object_temp.y_val,angular_vel_object_temp.z_val,
        linear_acc_object_temp.x_val,linear_acc_object_temp.y_val,linear_acc_object_temp.z_val,angular_acc_object_temp.x_val,angular_acc_object_temp.y_val,angular_acc_object_temp.z_val,
        enemy.position.x_val,enemy.position.y_val,enemy.position.z_val,enemy.linear_velocity.x_val,linear_vel_object_temp.y_val,linear_vel_object_temp.z_val
        ])
      
        collision_info = baseline_racer.airsim_client.simGetCollisionInfo()
        reward = compute_reward(pose_object_temp_numpy,linear_vel_object_temp, collision_info,pts)
        reward = np.array([reward])
        done = isDone(reward)
        if(done == 1):
            print("DONE ------ DONE \n")
        print('TEST ENVIROMENT - Action, Reward, Done:', action, reward, done,"\n")
        print("Remaining time in this test",timeout,"\n")
        state = next_state
        total_reward += reward
        timeout -= 1
print("Racing ended , reward is : ",total_reward)
