import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import torch.optim as optim

# Actor Neural Network
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc_units=400, fc1_units=300):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc1_units)
        self.fc3 = nn.Linear(fc1_units, action_size)

    def forward(self, state):
        """
        Build an actor (policy) network that maps states -> actions.
        Args:
            state: torch.Tensor with shape (batch_size, state_size)
        Returns:
            action: torch.Tensor with shape (batch_size, action_size)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.torch.tanh(self.fc3(x))

# Q1-Q2-Critic Neural Network  
  
class Critic(nn.Module):
    """
    Args:
        state_size: state dimension
        action_size: action dimension
        fc_units: number of neurons in one fully connected hidden layer
    """
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Q1 architecture
        self.l1 = nn.Linear(state_size + action_size, fc1_units)
        self.l2 = nn.Linear(fc1_units, fc2_units)
        self.l3 = nn.Linear(fc2_units, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_size + action_size, fc1_units)
        self.l5 = nn.Linear(fc1_units, fc2_units)
        self.l6 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        """
        Build a critic (value) network that maps state-action pairs -> Q-values.
        Args:
            state: torch.Tensor with shape (batch_size, state_size)
            action: torch.Tensor with shape (batch_size, action_size)
        Returns:
            x_1: torch.Tensor with shape (batch_size, 1)
            x_2: torch.Tensor with shape (batch_size, 1)
        """
        xa = torch.cat([state, action], 1)

        x1 = F.relu(self.l1(xa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xa))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)

        return x1, x2
    
    def Q1(self, state, action):
        xa = torch.cat([state, action], 1)
        x1 = F.relu(self.l1(xa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1

class Sys_R(nn.Module):
    def __init__(self,state_size, action_size, fc1_units=256, fc2_units=256):
        super(Sys_R, self).__init__()

        self.l1 = nn.Linear(2 * state_size + action_size, fc1_units)
        self.l2 = nn.Linear(fc1_units,fc2_units)
        self.l3 = nn.Linear(fc2_units, 1)


    def forward(self, state, next_state, action):
        xa = torch.cat([state, next_state, action], 1)

        x1 = F.relu(self.l1(xa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1

class SysModel(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=400, fc2_units=300):
        super(SysModel, self).__init__()
        self.l1 = nn.Linear(state_size + action_size, fc1_units)
        self.l2 = nn.Linear(fc1_units, fc2_units)
        self.l3 = nn.Linear(fc2_units, state_size)


    def forward(self, state, action):
        """Build a system model to predict the next state at given state and action.
        Returns:
            state: torch.Tensor with shape (batch_size, state_size)
        """
        xa = torch.cat([state, action], 1)

        x1 = F.relu(self.l1(xa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        return x1

"""
TD3_FORK AGENT
"""

class Agent:
    def __init__(
        self,
        load = False,
        gamma = 0.99, #discount factor
        lr_actor = 3e-4,  # learning rate for actor network
        lr_critic = 3e-4, # learning rate for critic network
        lr_sysmodel = 3e-4, # learning rate for system network
        lr_sysr = 3e-4, # learning rate for the system reward
        batch_size = 100,  # mini-batch size
        buffer_capacity = 1000000, # reply buffer capacitty
        tau = 0.02,  #target network update factor
        random_seed = np.random.randint(1,10000),  #random seed
        policy_noise=0.2,   # noise added to actor
        std_noise = 0.1,    # standard deviation for smoothing noise added to target policy
        noise_clip=0.5,   #noise bound
        policy_freq=2, #target network update period
        sys_weight=0.5
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make('BipedalWalkerHardcore-v3')
        self.create_actor()
        self.create_critic()
        self.create_sysmodel()
        self.create_sysr()
        self.act_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.crt_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.sys_opt = optim.Adam(self.sysmodel.parameters(), lr=lr_sysmodel) #define system model
        self.sysr_opt = optim.Adam(self.sysr.parameters(), lr=lr_sysr)
        self.set_weights()
        self.replay_memory_buffer = deque(maxlen = buffer_capacity)
        self.replay_memory_bufferd_dis = deque(maxlen = buffer_capacity)
        self.batch_size = batch_size
        self.tau = tau
        self.policy_freq = policy_freq
        self.gamma = gamma
        self.upper_bound = self.env.action_space.high[0] #action space upper bound
        self.lower_bound = self.env.action_space.low[0]  #action space lower bound
        self.obs_upper_bound = self.env.observation_space.high[0] #state space upper bound
        self.obs_lower_bound = self.env.observation_space.low[0]  #state space lower bound
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.std_noise = std_noise
        self.sys_weight = sys_weight
 

    # def create_actor(self):
    #     params = {
    #         'state_size':      self.env.observation_space.shape[0],
    #         'action_size':     self.env.action_space.shape[0],
    #         'seed':            88
    #     }
    #     self.actor = Actor(**params).to(self.device)
    #     self.actor_target = Actor(**params).to(self.device)

    def create_actor(self):
        params = {
            'state_size':      self.env.observation_space.shape[0],
            'action_size':     self.env.action_space.shape[0],
            'seed':            88
        }
        self.actor = Actor(**params).to(self.device)
        self.actor.load_state_dict(torch.load('actor.pth', map_location=torch.device('cpu'))) #a pre saved model
        self.actor_target = Actor(**params).to(self.device)
        self.actor_target.load_state_dict(torch.load('actor.pth', map_location=torch.device('cpu'))) #a pre saved model
        
    def create_critic(self):
        params = {
            'state_size':      self.env.observation_space.shape[0],
            'action_size':     self.env.action_space.shape[0],
            'seed':            88
        }
        self.critic = Critic(**params).to(self.device)
        self.critic_target = Critic(**params).to(self.device)

    def create_sysmodel(self):
        params = {
            'state_size':      self.env.observation_space.shape[0],
            'action_size':     self.env.action_space.shape[0]
        }
        self.sysmodel = SysModel(**params).to(self.device)
    
    def create_sysr(self):
        params = {
            'state_size':     self.env.observation_space.shape[0],
            'action_size':    self.env.action_space.shape[0]
        }
        self.sysr = Sys_R(**params).to(self.device)
            
    def set_weights(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


    def add_to_replay_memory(self, transition, buffername):
        #add samples to replay memory
        buffername.append(transition)

    def get_random_sample_from_replay_mem(self, buffername):
        #random samples from replay memory
        random_sample = random.sample(buffername, self.batch_size)
        return random_sample


    def learn_and_update_weights_by_replay(self, training_iterations, weight, totrain):
        """Update policy and value parameters using given batch of experience tuples.
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        weight is used for training  actor
        To train indicates whether we will train actor or not. We may not need to train Actor when the performance is good enough.
        This is a placeholder, you can define your criteria.
        """
        if len(self.replay_memory_buffer) < 1e4:
            return 1
        for it in range(training_iterations):
            mini_batch = self.get_random_sample_from_replay_mem(self.replay_memory_buffer)
            state_batch = torch.from_numpy(np.vstack([i[0] for i in mini_batch])).float().to(self.device)
            action_batch = torch.from_numpy(np.vstack([i[1] for i in mini_batch])).float().to(self.device)
            reward_batch = torch.from_numpy(np.vstack([i[2] for i in mini_batch])).float().to(self.device)
            next_state_batch = torch.from_numpy(np.vstack([i[3] for i in mini_batch])).float().to(self.device)
            done_list = torch.from_numpy(np.vstack([i[4] for i in mini_batch]).astype(np.uint8)).float().to(self.device)

            # Training and updating Actor & Critic networks.
            
            #Train Critic
            target_actions = self.actor_target(next_state_batch)
            offset_noises = torch.FloatTensor(action_batch.shape).data.normal_(0, self.policy_noise).to(self.device)

            #clip noise
            offset_noises = offset_noises.clamp(-self.noise_clip, self.noise_clip)
            target_actions = (target_actions + offset_noises).clamp(self.lower_bound, self.upper_bound)

            #Compute the target Q value
            Q_targets1, Q_targets2 = self.critic_target(next_state_batch, target_actions)
            Q_targets = torch.min(Q_targets1, Q_targets2)
            Q_targets = reward_batch + self.gamma * Q_targets * (1 - done_list)

            #Compute current Q estimates
            current_Q1, current_Q2 = self.critic(state_batch, action_batch)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, Q_targets.detach()) + F.mse_loss(current_Q2, Q_targets.detach())
            # Optimize the critic
            self.crt_opt.zero_grad()
            critic_loss.backward()
            self.crt_opt.step()

            # self.soft_update_target(self.critic, self.critic_target)


            """Train Sysmodel"""       
            next_state_prediction = self.sysmodel(state_batch, action_batch)
            next_state_prediction = next_state_prediction.clamp(self.obs_lower_bound, self.obs_upper_bound)
            sysmodel_loss = F.smooth_l1_loss(next_state_prediction, next_state_batch.detach())
            
            self.sys_opt.zero_grad()
            sysmodel_loss.backward()
            self.sys_opt.step()
            self.sysmodel_loss = sysmodel_loss.item()
            
            reward_prediction = self.sysr(state_batch, next_state_batch, action_batch)
            sysr_loss = F.mse_loss(reward_prediction, reward_batch.detach())
            self.sysr_opt.zero_grad()
            sysr_loss.backward()
            self.sysr_opt.step()
            self.sysr_loss = sysr_loss.item()
            
            s_flag = 1 if sysmodel_loss.item() < 0.010 else 0
            
            if it % self.policy_freq == 0:
                
                actor_loss1 = -self.critic.Q1(state_batch, self.actor(state_batch)).mean()
                
                if s_flag == 1:
                    p_next_state = self.sysmodel(state_batch, self.actor(state_batch))
                    p_next_state = p_next_state.clamp(self.obs_lower_bound,self.obs_upper_bound)
                    actions2 = self.actor(p_next_state.detach())
                    p_next_r = self.sysr(state_batch, p_next_state.detach(), self.actor(state_batch))

                    p_next_state2 = self.sysmodel(p_next_state, self.actor(p_next_state.detach()))
                    p_next_state2 = p_next_state2.clamp(self.obs_lower_bound, self.obs_upper_bound)
                    p_next_r2 = self.sysr(p_next_state.detach(), p_next_state2.detach(), self.actor(p_next_state.detach()))
                    actions3 = self.actor(p_next_state2.detach())

                    actor_loss2 =  self.critic.Q1(p_next_state2.detach(), actions3)
                    actor_loss3 =  -(p_next_r + self.gamma * p_next_r2 + self.gamma ** 2 * actor_loss2).mean()
                    actor_loss = (actor_loss1 + self.sys_weight * actor_loss3)
                else:
                    actor_loss = actor_loss1
                
                self.crt_opt.zero_grad()
                self.sys_opt.zero_grad()
                self.act_opt.zero_grad()
                actor_loss.backward()
                self.act_opt.step()
                
                # Update the frozen target models
                self.soft_update_target(self.critic, self.critic_target)
                self.soft_update_target(self.actor, self.actor_target)                
            
                

    def soft_update_target(self,local_model,target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def policy(self,state):
        """select action based on ACTOR but with noise added"""
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(state).cpu().data.numpy()
        self.actor.train()
        # Adding noise to action
        shift_action = np.random.normal(0, self.std_noise, size=self.env.action_space.shape[0])
        sampled_actions = (actions + shift_action)
        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions,self.lower_bound,self.upper_bound)
        return np.squeeze(legal_action)


    def select_action(self,state):
        """select action based on ACTOR"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            actions = self.actor_target(state).cpu().data.numpy()
        return np.squeeze(actions)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.crt_opt.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.act_opt.state_dict(), filename + "_actor_optimizer")

        torch.save(self.sysmodel.state_dict(), filename + "_sysmodel")
        torch.save(self.sys_opt.state_dict(), filename + "_sysmodel_optimizer")

        torch.save(self.sysr.state_dict(), filename + "_reward_model")
        torch.save(self.sysr_opt.state_dict(), filename + "_reward_model_optimizer")
        
        torch.save(self.replay_memory_buffer, filename + "_replay_memory_buffer")
        torch.save(self.replay_memory_bufferd_dis, filename + "replay_memory_bufferd_dis")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
        self.crt_opt.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.act_opt.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

        self.sysmodel.load_state_dict(torch.load(filename + "_sysmodel.pth"))
        relf.sys_opt.load_state_dict(torch.load(filename + "_sysmodel_optimizer"))

        self.sysr.load_state_dict(torch.load(filename + "_reward_model.pth"))
        relf.sysr_opt.load_state_dict(torch.load(filename + "_reward_model_optimizer"))
        
        self.replay_memory_buffer = torch.load(filename + "_replay_memory_buffer")
        self.replay_memory_bufferd_dis = torch.load(filename + "replay_memory_bufferd_dis")