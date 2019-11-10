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
#import gym, gym.spaces, gym.utils, gym.utils.seeding
import math
from tensorboardX import SummaryWriter # using log kinda of thing? , though need to check for ctrl+c early crash situtations dont save yet.
import os # for saving model file
import glob # for finding latest file in checkpoints folrdr



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
PPO_STEPS           = 1024  #256
MINI_BATCH_SIZE     = 256 #64 i guess this was supposed to be 256/64 = 4 ? ppo_iter like frame our state is 25 element.
PPO_EPOCHS          = 10
TEST_EPOCHS         = 10
NUM_TESTS           = 10
TARGET_REWARD       = 2500
LOAD_MODEL = False

currentGateIndexItShouldMoveFor = 0
distance_last_step = 0
baseline_racer = BaselineRacer(drone_name="drone_1",viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0])
#baseline_racer2 = BaselineRacer("drone_2",viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0])


#baseline_racer2.initialize_drone()

#start_position = baseline_racer.airsim_client.simGetVehiclePose(vehicle_name="drone_1").position
#gitbakalim = airsim.Vector3r(start_position.x_val+5, start_position.y_val+5, start_position.z_val+5)

#baseline_racer2.takeoffAsync()

#baseline_racer.airsim_client.moveToPositionAsync(5,5,5,3)
#baseline_racer.airsim_client.moveToZAsync(10.0,5)
#baseline_racer.airsim_client.moveToPositionAsync(start_position.x_val+15,start_position.y_val+15,start_position.z_val+15,32)

initX = -.55265
initY = -31.9786
initZ = -19.0225
#print(baseline_racer.airsim_client.simGetVehiclePose())

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
#baseline_racer.get_ground_truth_gate_poses()

#baseline_racer.airsim_client.moveToPositionAsync(8.887084007263184,18.478761672973633,2.0199999809265137, 5).join()

def isPassedGate(posOfDrone,middleCoordinatesGate,threesholdDensityGate = 0.5):
    #assuming Z is stable and same? 
    #maybe later on do all corners rather than just middle x+,x- and some threehold 
    #not even sure 0.3 is correct just testing if? 
    # maybe later can calculate via some testing on move functions and distance

    right_x = middleCoordinatesGate[0] + 0.5
    left_x = middleCoordinatesGate[0] -0.5
    up_y = middleCoordinatesGate[1] + 0.5
    down_y = middleCoordinatesGate[1] -0.5

    if( left_x <= posOfDrone[0] and right_x >= posOfDrone[0] and down_y <= posOfDrone[1] and up_y >= posOfDrone[1]):
        return True
    return False


#Number of Actions explained
#takes baseline_racer object not client object
def returnGateLocationsNumpy(obje_of_racer):
    gate_names_sorted_bad = sorted(obje_of_racer.airsim_client.simListSceneObjects("Gate.*"))
    # In building 99, for example, gate_names_sorted_bad would be ['Gate0', 'Gate10_21', 'Gate11_23', 'Gate1_3', 'Gate2_5', 'Gate3_7', 'Gate4_9', 'Gate5_11', 'Gate6_13', 'Gate7_15', 'Gate8_17', 'Gate9_19']
    # number after underscore is unreal garbage. and leading zeros are missing, so the following lines fix the same
    gate_indices_bad = [int(gate_name.split('_')[0][4:]) for gate_name in gate_names_sorted_bad]
    gate_indices_correct = sorted(range(len(gate_indices_bad)), key=lambda k:gate_indices_bad[k])
    gate_names_sorted = [gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct]
    gate_poses = [obje_of_racer.airsim_client.simGetObjectPose(gate_name) for gate_name in gate_names_sorted]
    npArray = np.ones((len(gate_poses),3),dtype=np.float32)
    for i in range(0, len(gate_poses)-1):
        npArray[i] = np.array([gate_poses[i].position.x_val,gate_poses[i].position.y_val,gate_poses[i].position.z_val])
        

    return npArray
        
"""
ACTIONS SHOULD BE CONTINOUS INSTEAD OF SCALING FACTOR ? just use nn output
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
def compute_reward(current_position,current_linear_velocity, collision_info,pts):
 #TIME OUT lazim
    thresh_dist = 5
    minimum_dist = 0.5
    beta = 1
    reward = 0
    global currentGateIndexItShouldMoveFor
    global distance_last_step
    #now we dont know how to know if we passed gate* 
    # maybe there is gate passed function
    # else or also we need to get distance to gate so more rewards + reward
    # things i still didnt use like time,acc, other drone distance behind and forward ,etc 

    if collision_info.has_collided:
        reward = -1 #this is very bad ! disqualification .....
   
    dist = 1000

    minGate = 0
    distCurrent = dist

    if(currentGateIndexItShouldMoveFor  == len(pts)-1):
        #means all gates are removed?
        #Control if index are correct -1 ? is correct
        print("All gates passed? \n")
        #BEFORE REMOVING GATES does we remove just if it passes ?or like wait for a few rewards to get that high reward then remove it? like maybe its negatively or
        # times (10 times to get positive)
        return 100 
    for i in range(0, len(pts)-1):
        dist = min(dist, np.linalg.norm(np.cross((current_position - pts[i]), (current_position - pts[i+1])))/np.linalg.norm(pts[i]-pts[i+1]))
        if(dist < distCurrent):
            distCurrent = dist
            minGate = i

    #dist = min(dist, np.linalg.norm(np.cross((current_position - pts[0]), (current_position - pts[0+1])))/np.linalg.norm(pts[0]-pts[0+1]))

    print("Distance the closest gate and its distance = ",minGate,dist,"\n")

    if(currentGateIndexItShouldMoveFor != minGate and ( abs(currentGateIndexItShouldMoveFor-minGate) > 1 )):
        reward = -1
        print("Oh shit , moving wrong gate direction ! I should be going ",currentGateIndexItShouldMoveFor,"but im going",minGate,"\n")
        # Moving direction of wrong gate shit thing , it shouldnt go !
    elif distCurrent < minimum_dist: #means passed gate
        reward = 10
        print("If deletion of Gate is going to be it?\n")
        #reward += 100 #150 total?
        print("before gate it should go deletion and len",currentGateIndexItShouldMoveFor,"\n")
        #pts = np.delete(pts,minGate)
        currentGateIndexItShouldMoveFor += 1
        distance_last_step = distCurrent
        print("after gate it should go deletion and len",currentGateIndexItShouldMoveFor,"\n")

        """if(isPassedGate(current_position,pts[minGate]) == True):
            print("If deletion of Gate is going to be it?\n")
            reward += 100 #150 total?
            print("before gate deletion and len",pts,len(pts),"\n")
            #pts = np.delete(pts,minGate)
            currentGateIndexItShouldMoveFor += 1
            print("after gate deletion and len",pts,len(pts),"\n")
            #global disari cikiyor mu"""
    
    elif distance_last_step > distCurrent:
        #it get closed to target gate
        print("it get closed to target gate","\n")
        reward = 3 
        if(distCurrent < thresh_dist):
            reward  += 1
    elif distCurrent < thresh_dist:
           reward = 1
    """else:
        print("Distance is smaller than threst distance but not good as minimum_dist \n")
        reward_dist = (math.exp(-beta*dist) - 0.5) 
        reward_speed = (np.linalg.norm([current_linear_velocity.x_val, current_linear_velocity.y_val, current_linear_velocity.z_val]) - 0.5)
        reward = reward_dist + reward_speed"""
    distance_last_step = distCurrent        
    print("Current Reward",reward,"\n")
    return float(reward)



def compute_gae(next_value, rewards, masks, values, gamma=GAMMA, lam=GAE_LAMBDA):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])

    return returns




"""
    position = Vector3r()3
    orientation = Quaternionr()
    linear_velocity = Vector3r() 3
    angular_velocity = Vector3r() 3
    dusman POZISYON 3
    Dusman angular 
    dusman linear VELO 3
    linear_acceleration = Vector3r()
    angular_acceleration = Vector3r(),

    For now only going to use linear_velocity hence i dont know much about others?

"""
def ppo_iter(states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    #print("hey this is probably the wrong part, whats is resultf of this \n")
    #print(batch_size,"--- // ",MINI_BATCH_SIZE,"\n")
    # generates random mini-batches until we have covered the full batch
    for _ in range(batch_size // MINI_BATCH_SIZE):
        rand_ids = np.random.randint(0, batch_size, MINI_BATCH_SIZE)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
 



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
            #print("Count steps for debug \n",count_steps)

    
    #print("Count steps for debug \n",count_steps)
    writer.add_scalar("returns", sum_returns / count_steps, frame_idx)
    writer.add_scalar("advantage", sum_advantage / count_steps, frame_idx)
    writer.add_scalar("loss_actor", sum_loss_actor / count_steps, frame_idx)
    writer.add_scalar("loss_critic", sum_loss_critic / count_steps, frame_idx)
    writer.add_scalar("entropy", sum_entropy / count_steps, frame_idx)
    writer.add_scalar("loss_total", sum_loss_total / count_steps, frame_idx)

"""

 
#stonegodYesterday at 11:44 PM
if( bb.ix <= p.x && p.x <= bb.ax && bb.iy <= p.y && p.y <= bb.ay ) {
    // Point is in bounding box
}
ax = middle_point + (width/2)
ix = middle_point - (width / 2)
iy = middle_point.y + (height / 2)
ay = middle_point.y - (height / 2)
"""





def isDone(reward):
    done = 0
    if  reward <= -1:
        done = 1
    elif reward >= 100:
        done = 1
        ## develop more.
   
    return done

def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x

def test_env(baseline_racer, model, device,testIndex, deterministic=True):
    #env needs to be resetted
    print("Resetted Again to start ",testIndex,"test.\n")
    timeout = 500 # 7 min (420sec)  / 0.1(sec one move) = didnt work it yet so made it 42000   other drone do it like 60k
    baseline_racer.reset_race()
    baseline_racer.start_race(1)
    #baseline_racer.start_odometry_callback_thread()
   
    #baseline_racer.takeoffAsync()
    baseline_racer.takeoff_with_moveOnSpline()
    state = np.ones(number_of_inputs)
    done = 0
    total_reward = 0
    global currentGateIndexItShouldMoveFor
    global distance_last_step


    currentGateIndexItShouldMoveFor = 0
    while not done and timeout>= 0:
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        dist, _ = model(state)
        action = torch.clamp(dist.sample(), -1.0, 1.0)

        """action = torch.clamp(dist.mean(),-1.0,1.0) if deterministic \
            else torch.clamp(dist.sample(),-1.0,1.0)"""
        if(deterministic == True):
            action = dist.mean.detach().cpu().numpy()[0]
            actx = action
            baseline_racer.airsim_client.moveByVelocityAsync(quad_vel.x_val+ actx[0], quad_vel.y_val+actx[1], quad_vel.z_val+actx[2],0.3).join()

        else:
            action = torch.clamp(dist.sample(),-1.0,1.0)
            actx = action.data.cpu().numpy()
            baseline_racer.airsim_client.moveByVelocityAsync(quad_vel.x_val+ actx[0][0], quad_vel.y_val+actx[0][1], quad_vel.z_val+actx[0][2],0.3).join()




        #time.sleep(0.5) # how much to sleep?
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
    #baseline_racer.stop_odometry_callback_thread()    
    return total_reward

#creating of input something
number_of_inputs = 21 # for now its beta phase , linear velocity of our drone and enemy drone, position of our drone and enemy drone 3+3+3+3=12 enemy 6, our 15 now
number_of_actions = 3 # it was 7 but x,y,z makes more sense than x+,x- etc. hovering action is not neeeded for now?
# also Timing could be a output that later on but now drone will move 1second fixed or 0.5 second or 1.5 second 
model = ActorCritic(number_of_inputs,number_of_actions,HIDDEN_SIZE).to(device)
if(LOAD_MODEL == True):
    findListedFile = glob.glob(os.getcwd()+"/checkpoints/*")
    latest_model_file_name = max(findListedFile,key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model_file_name))
print(model)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


train_epoch = 0
best_reward = None
frame_idx  = 0
early_stop = False

# needed for gym not really here.stateLimits = np.inf*np.ones([number_of_inputs])

# State computation
# our info (LINVELXYZ+OURXYZ ) + enemy info(LINVELXYZ+XYZ)
state = np.ones(number_of_inputs)


## starting
baseline_racer.load_level("Soccer_Field_Easy")
baseline_racer.initialize_drone()
baseline_racer.start_odometry_callback_thread()
baseline_racer.start_image_callback_thread()

pts = returnGateLocationsNumpy(baseline_racer)

#baseline_racer.start_race(3)
#baseline_racer.takeoffAsync()
writer = SummaryWriter(comment="ppo_" + "args.name") # index of each training. args.name args = parser.parse_args()





# ENV Reset function cagirilmalixd
while not early_stop:
    baseline_racer.reset_race()
   
    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    #state = np.zeros(number_of_inputs) ## our x,y,z position, our x,y,z linear vel , enemy xyz pos and vel
    #state = gym.spaces.box(low=-stateLimits,high=stateLimits,dtype=float32)
    

    #pose_object_temp = baseline_racer.airsim_client.getMultirotorState().kinematics_estimated.position
    #linear_vel_object_temp = baseline_racer.airsim_client.getMultirotorState().kinematics_estimated.linear_velocity

    #maybe time.sleep here for 0.1?
    baseline_racer.start_race(1)
    #baseline_racer.takeoffAsync()
    baseline_racer.takeoff_with_moveOnSpline()
    currentGateIndexItShouldMoveFor = 0


    for stepNumber in range(PPO_STEPS):
        print("this is ",stepNumber,"the state \n")
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        #view(-1,1) of tensor to not mess dimensions.
        dist, value = model(state)

        ##action = dist.sample()
        action = torch.clamp(dist.sample(), -1.0, 1.0)

        # each state, reward, done is a list of results from each parallel environment

        #quad_offset = interpret_action(actions)
        # Clipping the action about what? which parameters would be good
        # LET just say 1 second for now?
       
        quad_vel = baseline_racer.current_linear_velocity  #.getMultirotorState().kinematics_estimated.linear_velocity
        actx = action.data.cpu().numpy()
        baseline_racer.airsim_client.moveByVelocityAsync(quad_vel.x_val+ actx[0][0], quad_vel.y_val+actx[0][1], quad_vel.z_val+actx[0][2],0.3).join()
        #time.sleep(0.5) # how much to sleep?

        """#THIS IS WRONG ------------------------- next_state = np.zeros(number_of_inputs)
         ## our x,y,z position, our x,y,z linear vel , enemy xyz pos and vel

        pose_object_temp = baseline_racer.airsim_client.getMultirotorState().kinematics_estimated.position
        pose_object_temp_numpy = np.array([pose_object_temp.x_val,pose_object_temp.y_val,pose_object_temp.z_val])

        linear_vel_object_temp = baseline_racer.airsim_client.getMultirotorState().kinematics_estimated.linear_velocity
       # linear_vel_object_temp_numpy = np.array([linear_vel_object_temp.x_val,linear_vel_object_temp.y_val,linear_vel_object_temp.z_val])

        angular_vel_object_temp = baseline_racer.airsim_client.getMultirotorState().kinematics_estimated.angular_velocity
        linear_acc_object_temp = baseline_racer.airsim_client.getMultirotorState().kinematics_estimated.linear_acceleration
        angular_acc_object_temp = baseline_racer.airsim_client.getMultirotorState().kinematics_estimated.angular_acceleration
        enemy = baseline_racer.airsim_client.simGetGroundTruthKinematics("drone_2")"""

        pose_object_temp = baseline_racer.current_position
        pose_object_temp_numpy = np.array([pose_object_temp.x_val,pose_object_temp.y_val,pose_object_temp.z_val])

        linear_vel_object_temp = baseline_racer.current_linear_velocity
        angular_vel_object_temp = baseline_racer.current_angular_velocity
        linear_acc_object_temp = baseline_racer.current_linear_acceleration
        angular_acc_object_temp = baseline_racer.current_angular_acceleration

        enemy = baseline_racer.current_enemy

        # HIGH PROBABILTY just using [pose_object_temp] +  does this already but later will try or cat [] this saw on somewhere whileloking
        # maybe not because these arent numpy lolz




        next_state = np.array([pose_object_temp.x_val,pose_object_temp.y_val,pose_object_temp.z_val,linear_vel_object_temp.x_val,
        linear_vel_object_temp.y_val,linear_vel_object_temp.z_val,angular_vel_object_temp.x_val,angular_vel_object_temp.y_val,angular_vel_object_temp.z_val,
        linear_acc_object_temp.x_val,linear_acc_object_temp.y_val,linear_acc_object_temp.z_val,angular_acc_object_temp.x_val,angular_acc_object_temp.y_val,angular_acc_object_temp.z_val,
        enemy.position.x_val,enemy.position.y_val,enemy.position.z_val,enemy.linear_velocity.x_val,linear_vel_object_temp.y_val,linear_vel_object_temp.z_val
        ])

        #reward =np.array([-5]) # compute_reward(next_state, quad_vel, collision_info)
        collision_info = baseline_racer.airsim_client.simGetCollisionInfo()
        reward = compute_reward(pose_object_temp_numpy,linear_vel_object_temp, collision_info,pts)
        reward = np.array([reward])
        done = isDone(reward)
        print('Action, Reward, Done:', action, reward, done)



        # lets say we do this above ? next_state, reward, done, _ = envs.step(action.cpu().numpy())
        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)
        values.append(value)
        #reward = torch.from_numpy(np.array(reward))
        #reward = torch.as_tensor(reward)
        #print("Done*****value,",done,"\n")
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(1-done)#masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            
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
    #baseline_racer.stop_odometry_callback_thread()

    if train_epoch % TEST_EPOCHS == 0:
        test_reward = np.mean([test_env(baseline_racer, model, device,testIndex) for testIndex in range(NUM_TESTS)])
        writer.add_scalar("test_rewards", test_reward, frame_idx)
        print('Frame %s. reward: %s' % (frame_idx, test_reward))
            # Save a checkpoint every time we achieve a best reward
        if best_reward is None or best_reward < test_reward:
            if best_reward is not None:
                print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
                name = "%s_best_%+.3f_%d.dat" % ("args.name", test_reward, frame_idx)
                fname = os.path.join('.', 'checkpoints', name)
                torch.save(model.state_dict(), fname)
            best_reward = test_reward
        if test_reward > TARGET_REWARD: early_stop = True
