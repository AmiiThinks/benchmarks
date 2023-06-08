import random

import torch
from torch.nn import functional as F

from utils import one_hot_cross_entropy


def sample_dyna_buffer(buffer, n):
  n_samples = min(len(buffer), n)
  sample_hist = random.sample(list(buffer), n_samples)
  sample_obs = torch.stack([obs._tensor for obs, _ in sample_hist])
  sample_acts = torch.stack([act._tensor for _, act in sample_hist])
  return sample_obs, sample_acts

def get_critic_loss(
  qnet, target_qnet, obs, next_obs, acts, rewards, gammas,
  trans_model=None, rollout_len=1, input_noise_mean=0, input_noise_std=0):
  return get_dyna_loss(
    obs, acts, qnet, target_qnet, trans_model, next_obs,
    rewards, gammas).mean()

def get_dyna_loss(obs, acts, qnet, target_qnet, trans_model=None,
                  next_obs=None, rewards=None, gammas=None, next_states=None,
                  include_encoder_loss=False, return_step_data=False,
                  input_noise_mean=0, input_noise_std=0, rollout_len=1):

  assert rollout_len == 1 or trans_model is not None, \
    'trans_model must be provided for rollout_len > 1!'

  sample_states = qnet.encode(obs)
  if not include_encoder_loss:
    sample_states = sample_states.detach()
  
  if input_noise_std > 0 or input_noise_mean > 0:
    sample_states = sample_states + torch.randn_like(sample_states) \
      * input_noise_std + input_noise_mean

  q_vals = qnet.forward_encoded(sample_states)
  q_vals = q_vals.gather(1, acts.unsqueeze(1)).squeeze(1)

  with torch.no_grad():
    if trans_model:
      # Get initial state
      new_states = sample_states.detach()

      rollout_states = []
      rollout_rewards = []
      rollout_gammas = []

      ### Rollout model several steps ###

      for _ in range(rollout_len):
        new_acts = target_qnet.forward_encoded(new_states).argmax(dim=-1)
        new_states, new_rewards, new_gammas = trans_model(new_states, new_acts)
        new_gammas[new_gammas < 0.01] = 0 # Allow for hard termination

        rollout_states.append(new_states)
        rollout_rewards.append(new_rewards.squeeze(1))
        rollout_gammas.append(new_gammas.squeeze(1))

      # Start: a
      # [b, TERM]
      # [0, 1]
      # [0.99, 0]

      ### Calculate target returns ###

      # Bootstrap final value
      final_acts = target_qnet.forward_encoded(new_states).argmax(dim=-1)
      final_q_vals = qnet.forward_encoded(new_states)
      final_q_vals = final_q_vals.gather(1, final_acts.unsqueeze(1)).squeeze(1)

      rollout_rewards = rollout_rewards + [final_q_vals]
      rollout_gammas = [1] + rollout_gammas

      # Start: a
      # [b, TERM]
      # [0, 1, 0?]
      # [1, 0.99, 0]
      #
      # 0 * (0? + 0) = 0
      # 0.99 * (1 + 0) = 0.99
      # 1 * (0 + 0.99) = 0.99
      # 1 * (0 + 0.99 * (1 + 0 * (0? + 0)))

      # Iterate backwards over rewards and gammas
      target_q = torch.zeros_like(rollout_rewards[-1])
      for reward, gamma in zip(reversed(rollout_rewards), reversed(rollout_gammas)):
        target_q = gamma * (reward + target_q)

    elif next_obs is not None:
      next_states = qnet.encode(next_obs)
      next_acts = target_qnet.forward_encoded(next_states).argmax(dim=-1)
      next_q_vals = qnet.forward_encoded(next_states)
      next_q_vals = next_q_vals.gather(1, next_acts.unsqueeze(1)).squeeze(1)
      target_q = rewards + gammas * next_q_vals
      
  dyna_losses = F.mse_loss(q_vals, target_q, reduction='none')

  if return_step_data:
    return dyna_losses, (sample_states, acts, next_states, rewards, gammas)
  return dyna_losses

def get_transition_losses(batch_data, qnet, target_qnet, trans_model, args, encode=None):
  with torch.no_grad():
    next_states = qnet.encode(batch_data['next_obs'])

  oh_outcomes = None
  if args.stochastic == 'categorical':
    oh_outcomes, outcome_logits = trans_model.discretize(
      next_states, return_logits=True)
    
  next_state_pred, reward_pred, gamma_pred, stoch_logits = trans_model(
    batch_data['states'], batch_data['acts'],
    oh_outcomes=oh_outcomes, return_stoch_logits=True)

  trans_losses = {}
  if args.stochastic == 'categorical':
    stoch_probs = F.softmax(stoch_logits, dim=1)
    outcome_loss = one_hot_cross_entropy(stoch_probs, oh_outcomes.detach())
    trans_losses[f'outcome_loss'] = outcome_loss.mean()

  # Calculate the MSE losses

  # Get value equivalent loss instead of state loss if enabled
  if args.value_eq:
    rand_acts = torch.randint_like(
      batch_data['acts'], 0, qnet.n_acts, device=batch_data['acts'].device)
    next2_state_pred, reward2_pred, gamma2_pred = trans_model(
      next_state_pred.detach(), rand_acts)

    with torch.no_grad():
      next2_acts = target_qnet.forward_encoded(next2_state_pred).argmax(dim=-1)
      next2_q_vals = qnet.forward_encoded(next2_state_pred)
      next2_q_vals = next2_q_vals.gather(1, next2_acts.unsqueeze(1)).squeeze(1)
      target_vals = reward2_pred + gamma2_pred * next2_q_vals

    next_q_vals = qnet.forward_encoded(next_state_pred)
    next_q_vals = next_q_vals.gather(1, rand_acts.unsqueeze(1)).squeeze(1)

    trans_losses[f'value_eq_loss'] = F.mse_loss(
      next_q_vals, target_vals.detach(), reduction='mean')
  else:
    trans_losses[f'state_loss'] = F.mse_loss(
      next_state_pred, next_states.reshape(next_state_pred.shape).detach(), reduction='mean')
  trans_losses[f'reward_loss'] = F.mse_loss(
    reward_pred.squeeze(), batch_data['rewards'], reduction='mean')
  trans_losses[f'gamma_loss'] = F.mse_loss(
    gamma_pred.squeeze(), batch_data['gammas'], reduction='mean')

  return trans_losses