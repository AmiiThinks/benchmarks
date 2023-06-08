from collections import defaultdict
import time

from tqdm import tqdm

from env_helpers import *
from training_helpers import *
from model_construction import *
from utils import as_hashable

from dyna_utils import *
from dyna import *
from data import ReplayBuffer


LAST_TIME = time.time()
TASK_TIMES = defaultdict(int)


def get_time_split():
  global LAST_TIME
  split = time.time() - LAST_TIME
  LAST_TIME = time.time()
  return split


def train_dyna_model(args, encoder_model=None, trans_model=None):
  if args.wandb:
    global wandb
    import wandb

  env = make_env(args.env_name)
  env = FreezeOnDoneWrapper(env, max_count=1)
  act_space = env.action_space
  act_dim = act_space.n
  sample_obs = env.reset()
  sample_obs = preprocess_obs([sample_obs])

  freeze_encoder = False # not (args.ae_recon_loss or args.e2e_loss)

  # Load the encoder
  if encoder_model is None:
    ae_model, ae_trainer = construct_ae_model(
      sample_obs.shape[1:], args)
    if ae_trainer is not None:
      ae_trainer.log_freq = -1
    # encoder_model = ae_model.get_encoder()
  else:
    ae_model = encoder_model

  if freeze_encoder:
    freeze_model(ae_model)
    ae_model.eval()
  else:
    ae_model = ae_model.to(args.device)
    ae_model.train()
  # print(f'Loaded encoder')

  use_er = (args.er_model or args.er_train_model) and args.n_dyna_updates > 0
  if use_er:
    replay_buffer = ReplayBuffer(args.replay_size)
  use_trans_model = (not args.er_model) and args.n_dyna_updates > 0

  # Load the transition model
  if use_trans_model:
    if trans_model is None:
      # Use float inputs even for discrete models
      args.use_soft_embeds = True
      trans_model = construct_trans_model(
      ae_model, args, act_space, load=False)[0]
    trans_model = trans_model.to(args.device)
    trans_model.train()
    # print(f'Loaded transition model')
  else:
    trans_model = None

  if args.wandb:
    wandb.config.update(args, allow_val_change=True)

  qnet_params = dict(
    obs_dim = sample_obs.shape[1:],
    n_acts = act_dim,
    ae = ae_model.cpu(),
    hidden_sizes = args.q_hidden)
  qnet = DuelingDQNModel(**qnet_params)
  target_qnet = DuelingDQNModel(**qnet_params)
  qnet.update_target(target_qnet)

  qnet = qnet.to(args.device)
  target_qnet = target_qnet.to(args.device)

  # Create the optimizer(s)

  all_params = list(qnet.parameters())
  all_params += list(ae_model.parameters())
  qnet_optimizer = optim.Adam(all_params, lr=args.learning_rate)

  if trans_model:
    trans_optimizer = optim.Adam(trans_model.parameters(), lr=args.learning_rate)

  if ae_trainer is not None:
    ae_trainer.optimizer = qnet_optimizer

  run_stats = defaultdict(list)

  # Dyna algorithm
  curr_obs = env.reset()
  curr_obs = torch.from_numpy(curr_obs).float()
  unique_hist = set()
  # Batch q_vals, next_obs, rewards, acts, gammas
  ep_rewards = []
  n_batches = int(np.ceil(args.dyna_steps / args.batch_size))
  step = 0
  for batch in tqdm(range(n_batches)):
    ae_model.eval()
    batch_data = {k: [] for k in ['obs', 'states', 'q_vals', 'next_obs', 'rewards', 'acts', 'gammas']}
    get_time_split()
    for _ in range(args.batch_size):
      with torch.no_grad():
        state = qnet.encode(curr_obs.unsqueeze(0).to(args.device))
        q_vals = qnet.forward_encoded(state)
      batch_data['obs'].append(curr_obs)
      batch_data['states'].append(state.squeeze(0))
      batch_data['q_vals'].append(q_vals.squeeze(0))

      # Epsilon-greedy sample an action
      if np.random.rand() < args.epsilon:
        act = torch.randint(qnet.n_acts, (1,))[0].long()
      else:
        act = q_vals.squeeze(0).argmax().cpu()
      batch_data['acts'].append(act)

      unique_hist.add((as_hashable(curr_obs), as_hashable(act)))

      # Take the action
      next_obs, reward, done, _ = env.step(act)
      next_obs = torch.from_numpy(next_obs).float()
      ep_rewards.append(reward)
      batch_data['next_obs'].append(next_obs)
      batch_data['rewards'].append(torch.tensor(reward).float())
      batch_data['gammas'].append(
        torch.tensor(args.gamma * (1 - done)).float())
      
      # Update replay buffer
      if use_er:
        replay_buffer.add_step(
          curr_obs, act, next_obs, batch_data['rewards'][-1], batch_data['gammas'][-1])

      # Update the current obs
      if done:
        replay_buffer.add_step(
          next_obs, act, next_obs, batch_data['rewards'][-1], batch_data['gammas'][-1])

        curr_obs = env.reset()
        curr_obs = torch.from_numpy(curr_obs).float()
        run_stats['ep_length'].append(len(ep_rewards))
        # print('\n--- Episode Stats ---')
        # print(f'Reward: {sum(ep_rewards)}')
        # print(f'Length: {len(ep_rewards)}')
        ep_rewards = []
      else:
        curr_obs = next_obs
      
      update_stats(run_stats, {
        'reward': reward,
        'q_value': q_vals.squeeze()[act].item()
      })

      # Target net update
      if step % args.target_net_update_freq == 0:
        qnet.update_target(target_qnet)
        # print(f'Updated target network at step {step}')

      # Log and reset logging data
      if step > 0 and step % args.log_freq == 0:
        log_stats(run_stats, step, args)
        run_stats = defaultdict(list)

      step += 1

    TASK_TIMES['Env Rollouts'] += get_time_split()
    
    ae_model.train()

    batch_data = {k: torch.stack(v).to(args.device) \
      for k, v in batch_data.items()}


    ### AE Reconstruction Loss ###


    if args.ae_recon_loss:
      get_time_split()
      loss_dict = ae_trainer.train((batch_data['obs'], _, batch_data['next_obs']))[0]
      TASK_TIMES['Autoencoder Training'] += get_time_split()
      for k, v in loss_dict.items():
        run_stats[k].append(v.item())


    ### Update the qnet ###


    if args.critic_input_noise:
      input_noise_std = 0.005
    else:
      input_noise_std = 0

    if not args.no_online_update:
      get_time_split()
      critic_loss = get_critic_loss(
        qnet, target_qnet, batch_data['obs'],
        batch_data['next_obs'], batch_data['acts'],
        batch_data['rewards'], batch_data['gammas'],
        input_noise_std=input_noise_std)
        # trans_model=trans_model, rollout_len=args.critic_n_step)

      qnet_optimizer.zero_grad()
      critic_loss.backward()
      qnet_optimizer.step()
      TASK_TIMES['On-Policy QNet Training'] += get_time_split()

      run_stats['critic_loss'].append(critic_loss.item())


    ### Transition model update ###


    if use_trans_model:
      get_time_split()
      for _ in range(args.n_trans_updates):
        if args.er_train_model:
          obs, acts, next_obs, rewards, gammas = \
            to_device(replay_buffer.sample(args.dyna_batch_size), args.device)
          states = qnet.encode(obs)
          if not args.e2e_loss:
            states = states.detach()
          trans_data = {
            'states': states, 'acts': acts, 'next_obs': next_obs,
            'rewards': rewards, 'gammas': gammas}
        else:
          trans_data = batch_data
        trans_data['acts'] = trans_data['acts']

        trans_losses = get_transition_losses(
          trans_data, qnet, target_qnet, trans_model, args)
        trans_loss = torch.sum(torch.stack(list(trans_losses.values())))

        trans_optimizer.zero_grad()
        trans_loss.backward()
        trans_optimizer.step()

        update_stats(run_stats, {k: v.item() for k, v in trans_losses.items()})
        run_stats['trans_loss'].append(trans_loss.item())
      TASK_TIMES['Transition Model Training'] += get_time_split()


    ### Dyna updates ###


    if step >= args.dyna_start_step:
      get_time_split()
      for _ in range(args.n_dyna_updates):
        # unique_hist, qnet, target_qnet, trans_model

        if use_trans_model:
          if args.dyna_on_policy:
            obs, acts = batch_data['obs'], batch_data['acts']
          else:
            obs, acts = to_device(sample_dyna_buffer(
              unique_hist, args.dyna_batch_size), args.device)
          dyna_losses, step_data = get_dyna_loss(
            obs, acts, qnet, target_qnet, trans_model,
            include_encoder_loss=True, return_step_data=True,
            input_noise_std=input_noise_std, rollout_len=args.critic_n_step)

        else:
          obs, acts, next_obs, rewards, gammas = \
            to_device(replay_buffer.sample(args.dyna_batch_size), args.device)
          dyna_losses = get_dyna_loss(
            obs, acts, qnet, target_qnet,
            next_obs=next_obs, rewards=rewards, gammas=gammas,
            include_encoder_loss=True, input_noise_std=input_noise_std)
        dyna_loss = dyna_losses.mean()

        qnet_optimizer.zero_grad()
        dyna_loss.backward()
        qnet_optimizer.step()
        
        run_stats['dyna_q_loss'].append(dyna_loss.item())
      TASK_TIMES['Model-Based QNet Training'] += get_time_split()

  return qnet

# Source: https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'


if __name__ == '__main__':
  # Parse args
  dyna_arg_parser = make_dyna_arg_parser()
  args = get_args(dyna_arg_parser)

  # This is required because wandb sweeps don't support binary flags
  if args.er_model_binary is not None:
    args.er_model = bool(args.er_model_binary)
  if args.er_train_model_binary is not None:
    args.er_train_model = bool(args.er_train_model_binary)
  if args.e2e_loss_binary is not None:
    args.e2e_loss = bool(args.e2e_loss_binary)
  if args.ae_recon_loss_binary is not None:
    args.ae_recon_loss = bool(args.ae_recon_loss_binary)
  if args.dyna_on_policy_binary is not None:
    args.dyna_on_policy = bool(args.dyna_on_policy_binary)
  if args.critic_input_noise_binary is not None:
    args.critic_input_noise = bool(args.critic_input_noise_binary)
  if args.no_online_update_binary is not None:
    args.no_online_update = bool(args.no_online_update_binary)

  assert not (args.trans_model_type == 'discrete' and args.ae_model_type == 'vqvae'), \
    'A VQVAE with a discrete trans model is not fully differentiable, use a soft_vqvae instead!'

  # Setup wandb
  if args.wandb:
    import wandb
    wandb.init(project='stochastic-dyna', config=args, tags=args.tags,
      settings=wandb.Settings(start_method='thread'), allow_val_change=True)
    args = wandb.config

  # Train and test the model
  start_time = time.time()
  model = train_dyna_model(args)
  train_time = time.time() - start_time

  print('\n')
  print(bcolors.BOLD + bcolors.HEADER + '-' * 10 + '  Benchmark Report  ' + '-' * 10 + bcolors.ENDC)
  
  print()
  device = torch.device(args.device)
  if device.type == 'cuda':
    device_name = torch.cuda.get_device_name(device)
  else:
    device_name = 'CPU'

  print('Rollout Device: ' + str(device) + ', ' + str(device_name))
  print('Training Device: ' + str(device) + ', ' + str(device_name))
  
  print()
  print('Total Run Time: ' + bcolors.BOLD + bcolors.OKGREEN + \
        '{:.3f}s'.format(train_time) + bcolors.ENDC)
  print('Total Env Steps:' + bcolors.BOLD + ' {}'.format(args.dyna_steps) + bcolors.ENDC)

  print()
  for key, val in TASK_TIMES.items():
    print(key + ': ' + bcolors.BOLD + '{:.3f}s'.format(val) + bcolors.ENDC)
