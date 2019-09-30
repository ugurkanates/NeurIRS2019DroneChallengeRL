#metestpy
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


baseline_racer = BaselineRacer(drone_name="drone_1",viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0])
baseline_racer2 = BaselineRacer("drone_2",viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0])

baseline_racer.load_level("Soccer_Field_Easy")
baseline_racer.start_race(3)
baseline_racer.initialize_drone()

#baseline_racer2.initialize_drone()

#start_position = baseline_racer.airsim_client.simGetVehiclePose(vehicle_name="drone_1").position
#gitbakalim = airsim.Vector3r(start_position.x_val+5, start_position.y_val+5, start_position.z_val+5)

baseline_racer.takeoffAsync()
baseline_racer2.takeoffAsync()

#baseline_racer.airsim_client.moveToPositionAsync(5,5,5,3)
#baseline_racer.airsim_client.moveToZAsync(10.0,5)
#baseline_racer.airsim_client.moveToPositionAsync(start_position.x_val+15,start_position.y_val+15,start_position.z_val+15,32)

initX = -.55265
initY = -31.9786
initZ = -19.0225
print(baseline_racer.airsim_client.simGetVehiclePose())

#baseline_racer.airsim_client.moveToPositionAsync(1.0,-2.0, 1.0, 5).join()
#time.sleep(0.5)

#i=0
"""while(i<100):
    print(baseline_racer.airsim_client.simGetCollisionInfo(),"\n")
    print(baseline_racer.airsim_client.simGetGroundTruthKinematics("drone_1"))
    print(baseline_racer.airsim_client.simGetVehiclePose())
    i=+1
"""
#print(baseline_racer.airsim_client.simGetVehiclePose())


#baseline_racer.airsim_client.moveToPositionAsync(8.887084007263184,18.478761672973633,2.0199999809265137, 5).join()
### DETECT device
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print('Device:', device)

# Make RL agent
#PARAMETERS


HIDDEN_SIZE         = 256
LEARNING_RATE       = 1e-4
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
PPO_EPSILON         = 0.2
CRITIC_DISCOUNT     = 0.5
ENTROPY_BETA        = 0.001
PPO_STEPS           = 256
MINI_BATCH_SIZE     = 64
PPO_EPOCHS          = 10
TEST_EPOCHS         = 10
NUM_TESTS           = 10
TARGET_REWARD       = 2500



#Number of Actions explained

"""
def interpret_action(action):
    scaling_factor = 0.25
    if action == 0:
        quad_offset = (0, 0, 0)
    elif action == 1:
        quad_offset = (scaling_factor, 0, 0)
    elif action == 2:
        quad_offset = (0, scaling_factor, 0)
    elif action == 3:
        quad_offset = (0, 0, scaling_factor)
    elif action == 4:
        quad_offset = (-scaling_factor, 0, 0)    
    elif action == 5:
        quad_offset = (0, -scaling_factor, 0)
    elif action == 6:
        quad_offset = (0, 0, -scaling_factor)
        basically moving on all x,y,z both negative and positive ways plus hovering action(nothing)
"""

# Number of Inputs explained

"""
position = Vector3r()
    orientation = Quaternionr()
    linear_velocity = Vector3r()
    angular_velocity = Vector3r()
    linear_acceleration = Vector3r()
    angular_acceleration = Vector3r(),

    For now only going to use linear_velocity hence i dont know much about others?

"""
#creating of input something
number_of_inputs = 12 # for now its beta phase , linear velocity of our drone and enemy drone, position of our drone and enemy drone 3+3+3+3=12
number_of_actions = 7
model = ActorCritic(number_of_inputs,number_of_actions,HIDDEN_SIZE).to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


train_epoch = 0
best_reward = None
frame_idx  = 0
early_stop = False
# State computation
# our info (LINVELXYZ+OURXYZ ) + enemy info(LINVELXYZ+XYZ)

# ENV Reset function cagirilmalixd
while not early_stop:
    
    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    state = np.zeros(number_of_inputs) ## our x,y,z position, our x,y,z linear vel , enemy xyz pos and vel
    pose_object_temp = baseline_racer.airsim_client.getMultirotorState().kinematics_estimated.position
    linear_vel_object_temp = baseline_racer.airsim_client.getMultirotorState().kinematics_estimated.linear_velocity
    state[0] = pose_object_temp.x_val
    state[1] = pose_object_temp.y_val
    state[2] = pose_object_temp.z_val
    state[3] = linear_vel_object_temp.x_val
    state[4] = linear_vel_object_temp.y_val
    state[5] = linear_vel_object_temp.z_val




    baseline_racer.airsim_client.getMultirotorState().kinematics_estimated.position
    for _ in range(PPO_STEPS):
        #should change ve3ctr data to .
        state = torch.FloatTensor(state).to(device)
        #view(-1,1) of tensor to not mess dimensions.
        dist, value = model(state)

        action = dist.sample()
        # each state, reward, done is a list of results from each parallel environment

        quad_offset = interpret_action(actions)
        quad_vel = baseline_racer.airsim_client.getMultirotorState().kinematics_estimated.linear_velocity
        baseline_racer.airsim_client.moveByVelocityAsync(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2], 5).join()
        time.sleep(0.5)

        next_state = np.zeros(number_of_inputs)
         ## our x,y,z position, our x,y,z linear vel , enemy xyz pos and vel
        pose_object_temp = baseline_racer.airsim_client.getMultirotorState().kinematics_estimated.position
        linear_vel_object_temp = baseline_racer.airsim_client.getMultirotorState().kinematics_estimated.linear_velocity
        next_state[0] = pose_object_temp.x_val
        next_state[1] = pose_object_temp.y_val
        next_state[2] = pose_object_temp.z_val
        next_state[3] = linear_vel_object_temp.x_val
        next_state[4] = linear_vel_object_temp.y_val
        next_state[5] = linear_vel_object_temp.z_val





        quad_vel = baseline_racer.airsim_client.getMultirotorState().kinematics_estimated.linear_velocity
        reward = compute_reward(quad_state, quad_vel, collision_info)
        done = isDone(reward)
        print('Action, Reward, Done:', action, reward, done)



        # lets say we do this above ? next_state, reward, done, _ = envs.step(action.cpu().numpy())
        log_prob = dist.log_prob(action)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            
        states.append(state)
        actions.append(action)
            
        state = next_state
        frame_idx += 1

    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    returns = compute_gae(next_value, rewards, masks, values)

    returns   = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states)
    actions   = torch.cat(actions)
    advantage = returns - values
    advantage = normalize(advantage)  

    ppo_update(frame_idx, states, actions, log_probs, returns, advantage)
    train_epoch += 1

    if train_epoch % TEST_EPOCHS == 0:
        test_reward = np.mean([test_env(env, model, device) for _ in range(NUM_TESTS)])
        writer.add_scalar("test_rewards", test_reward, frame_idx)
        print('Frame %s. reward: %s' % (frame_idx, test_reward))
            # Save a checkpoint every time we achieve a best reward
        if best_reward is None or best_reward < test_reward:
            if best_reward is not None:
                print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
                name = "%s_best_%+.3f_%d.dat" % (args.name, test_reward, frame_idx)
                fname = os.path.join('.', 'checkpoints', name)
                torch.save(model.state_dict(), fname)
            best_reward = test_reward
        if test_reward > TARGET_REWARD: early_stop = True

def ppo_iter(states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    # generates random mini-batches until we have covered the full batch
    for _ in range(batch_size // MINI_BATCH_SIZE):
        rand_ids = np.random.randint(0, batch_size, MINI_BATCH_SIZE)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
 


def compute_reward(quad_state, quad_vel, collision_info):
    thresh_dist = 7
    beta = 1

    z = -10
    pts = [np.array([-.55265, -31.9786, -19.0225]), np.array([48.59735, -63.3286, -60.07256]), np.array([193.5974, -55.0786, -46.32256]), np.array([369.2474, 35.32137, -62.5725]), np.array([541.3474, 143.6714, -32.07256])]

    quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))

    if collision_info.has_collided:
        reward = -100
    else:    
        dist = 10000000
        for i in range(0, len(pts)-1):
            dist = min(dist, np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i+1])))/np.linalg.norm(pts[i]-pts[i+1]))

        #print(dist)
        if dist > thresh_dist:
            reward = -10
        else:
            reward_dist = (math.exp(-beta*dist) - 0.5) 
            reward_speed = (np.linalg.norm([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val]) - 0.5)
            reward = reward_dist + reward_speed

    return reward

def ppo_update(frame_idx, states, actions, log_probs, returns, advantages, clip_param=PPO_EPSILON):
    count_steps = 0
    sum_returns = 0.0
    sum_advantage = 0.0
    sum_loss_actor = 0.0
    sum_loss_critic = 0.0
    sum_entropy = 0.0
    sum_loss_total = 0.0

    # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
    for _ in range(PPO_EPOCHS):
        # grabs random mini-batches several times until we have covered all data
        for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track statistics
            sum_returns += return_.mean()
            sum_advantage += advantage.mean()
            sum_loss_actor += actor_loss
            sum_loss_critic += critic_loss
            sum_loss_total += loss
            sum_entropy += entropy
            
            count_steps += 1
    
    #writer.add_scalar("returns", sum_returns / count_steps, frame_idx)
    #writer.add_scalar("advantage", sum_advantage / count_steps, frame_idx)
    #writer.add_scalar("loss_actor", sum_loss_actor / count_steps, frame_idx)
    #writer.add_scalar("loss_critic", sum_loss_critic / count_steps, frame_idx)
    #writer.add_scalar("entropy", sum_entropy / count_steps, frame_idx)
    #writer.add_scalar("loss_total", sum_loss_total / count_steps, frame_idx)



def isDone(reward):
    done = 0
    if  reward <= -10:
        done = 1
    return done

def interpret_action(action):
    scaling_factor = 0.25
    if action == 0:
        quad_offset = (0, 0, 0)
    elif action == 1:
        quad_offset = (scaling_factor, 0, 0)
    elif action == 2:
        quad_offset = (0, scaling_factor, 0)
    elif action == 3:
        quad_offset = (0, 0, scaling_factor)
    elif action == 4:
        quad_offset = (-scaling_factor, 0, 0)    
    elif action == 5:
        quad_offset = (0, -scaling_factor, 0)
    elif action == 6:
        quad_offset = (0, 0, -scaling_factor)
    
    return quad_offset



def compute_gae(next_value, rewards, masks, values, gamma=GAMMA, lam=GAE_LAMBDA):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        # prepend to get correct order back
        returns.insert(0, gae + values[step])
    return returns
