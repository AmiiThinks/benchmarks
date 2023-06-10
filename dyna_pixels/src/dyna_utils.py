import argparse

import numpy as np


def make_dyna_arg_parser():
  # These are the dyna-specific arguments only
  parser = argparse.ArgumentParser()
  parser.add_argument('--q_hidden', type=int, nargs='*', default=[512, 256, 256])
  parser.add_argument('--dyna_steps', type=int, default=100_000)
  parser.add_argument('--epsilon', type=float, default=0.15)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--n_dyna_updates', type=int, default=1)
  # If using replay buffer, diff data each batch, otherwise same data each batch
  parser.add_argument('--n_trans_updates', type=int, default=1)
  parser.add_argument('--dyna_batch_size', type=int, default=1)
  parser.add_argument('--dyna_start_step', type=int, default=0)
  parser.add_argument('--target_net_update_freq', type=int, default=5000)
  parser.add_argument('--replay_size', type=int, default=100_000)
  parser.add_argument('--cbp_replace_rate', type=float, default=1e-4)
  parser.add_argument('--critic_n_step', type=int, default=1,
    help='How many steps to rollout model for critic target estimation.')
  
  # Use value equivalence to train the transition model
  parser.add_argument('--value_eq', action='store_true')

  parser.add_argument('--ae_recon_loss', action='store_true')
  parser.add_argument('--ae_recon_loss_binary', type=int, default=None)

  parser.add_argument('--e2e_loss', action='store_true')
  parser.add_argument('--e2e_loss_binary', type=int, default=None)

  parser.add_argument('--er_model', action='store_true')
  parser.add_argument('--er_model_binary', type=int, default=None)

  parser.add_argument('--er_train_model', action='store_true')
  parser.add_argument('--er_train_model_binary', type=int, default=None)

  parser.add_argument('--dyna_on_policy', action='store_true')
  parser.add_argument('--dyna_on_policy_binary', type=int, default=None)

  parser.add_argument('--critic_input_noise', action='store_true')
  parser.add_argument('--critic_input_noise_binary', type=int, default=None)

  parser.add_argument('--no_online_update', action='store_true')
  parser.add_argument('--no_online_update_binary', type=int, default=None)

  parser.set_defaults(
    er_model=False, er_train_model=False, value_eq=False,
    ae_recon_loss=False, e2e_loss=False, dyna_on_policy=False,
    critic_input_noise=False, no_online_update=False, cbp=False)
 
  return parser



def epsilon_greedy_sample(model, obs, epsilon):
  """Samples an action from the model with epsilon-greedy exploration."""
  if np.random.rand() < epsilon:
    return np.random.randint(model.n_acts)
  else:
    return model.predict(obs)

def update_stats(stats, update_dict):
  for k, v in update_dict.items():
    stats[k].append(v)

def log_stats(stats, step, args):
  mean_stats = {k: np.mean(v) for k, v in stats.items()}

  # Create a pretty log string
  log_str = f'\n--- Step {step} ---\n'
  for i, (k, v) in enumerate(mean_stats.items()):
    log_str += f'{k}: {v:.3f}'
    if i < len(mean_stats) - 1:
      if i % 3 == 2:
        log_str += '\n'
      else:
        log_str += '  \t| '
  # print(log_str)

  if args.wandb:
    # Remove nans for Wandb
    import wandb
    mean_stats = {k: v for k, v in mean_stats.items() if not np.isnan(v)}
    mean_stats['step'] = step
    wandb.log(mean_stats)
  
def to_device(tensors, device):
  return [t.to(device) for t in tensors]