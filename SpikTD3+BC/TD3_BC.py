import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from spikingjelly.activation_based import neuron, functional, layer, surrogate, base


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NonSpikingLIFNode(neuron.LIFNode):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def single_step_forward(self, x: torch.Tensor):
		self.v_float_to_tensor(x)

		if self.training:
			self.neuronal_charge(x)
		else:
			if self.v_reset is None:
				if self.decay_input:
					self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
				else:
					self.v = self.neuronal_charge_no_decay_input_reset0(x, self.v, self.tau)

			else:
				if self.decay_input:
					self.v = self.neuronal_charge_decay_input(x, self.v, self.v_reset, self.tau)
				else:
					self.v = self.neuronal_charge_no_decay_input(x, self.v, self.v_reset, self.tau)



class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, T=16):
		super(Actor, self).__init__()


		self.actor = nn.Sequential(
			layer.Linear(state_dim, 256),
			neuron.LIFNode(),
			layer.Linear(256, 256),
			neuron.LIFNode(),
			layer.Linear(256, action_dim),
			NonSpikingLIFNode(tau=2.0)
		)

		
		self.max_action = max_action
		self.T = T

	def forward(self, state):
		for t in range(self.T):
			self.actor(state)
			# print('=============================================================')
			# print('state: ', state.shape, '\n', state)
			# print('self.actor(state)', self.actor(state))
			# print('t: ', t, '\n', 'self.actor[-1].v', self.actor[-1].v.shape, '\n', self.actor[-1].v)
		a = self.actor[-1].v
		# print('state : ', len(list(state)))
		# print('a : ', a)
		# print('self.actor[-1].v * self.max_action: ', self.max_action * a)
		# print('self.actor[-1].v * self.max_action * tanh: ', self.max_action * torch.tanh(a))
		return self.max_action * torch.tanh(a)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

class TD3_BC(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=2.5,
	):

		T = 1
		# T = 8
		# T = 16
		print('=' * 50)
		print(f'now T = {T}')
		print('=' * 50)
		self.actor = Actor(state_dim, action_dim, max_action, T).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)   
		# print('self.actor(state).cpu().data.numpy().flatten(): ', self.actor(state).cpu().data.numpy().flatten())
		choose_action = self.actor(state).cpu().data.numpy().flatten()
		functional.reset_net(self.actor)
		return choose_action   


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		# state : torch.Size([256, 11])  action : torch.Size([256, 3])  next_state : torch.Size([256, 11])
		# reward : torch.Size([256, 1])  not_done : torch.Size([256, 1])

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)
			functional.reset_net(self.actor_target)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			functional.reset_net(self.critic_target)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)
		# print('current_Q1', current_Q1)
		# print('current_Q2', current_Q2)
		# print('target_Q', target_Q)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		# print('critic_loss : ', critic_loss)
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor loss
			pi = self.actor(state)
			functional.reset_net(self.actor)
			Q = self.critic.Q1(state, pi)   #
			# functional.reset_net(self.critic)
			lmbda = self.alpha/Q.abs().mean().detach()

			actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action) 
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
