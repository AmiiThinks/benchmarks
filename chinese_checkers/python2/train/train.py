import os, sys
import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
import optax
import haiku as hk
import time

# import file from outside directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from wrappers import pywrapper as pw
from wrappers import ccwrapper as cw
from models import AZeroModel
from helpers import utils, log_helper

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

# required game states
cc = cw.CCheckers()
gamestate = cw.CCState()
l = cw.CCLocalRank12()

class Train():
    def __init__(self, config, self_play_iter):
        # get hyperparameters and constants from config file
        self.config = config
        self.batch_size = config["training"]["batch_size"]
        self.learning_rate = config["training"]["learning_rate"]
        self.epochs = config["training"]["epochs"]
        self.momentum = config["training"]["momentum"]
        self.weight_decay = config["training"]["weight_decay"]
        self.seed = config["training"]["seed"]
        self.logs_path = config["file"]["logs_path"]
        self.train_set_path = config["file"]["training_data_path"]
        self.trained_model_params_path = config["file"]["trained_model_params_path"]
        self.training_plots_path = config["file"]["training_plots_path"]
        self.total_time = 0

        # define self play iteration
        self.self_play_iter = self_play_iter

        # define logger
        self.logger = log_helper.setup_logger('train_log', self.logs_path, 'train.log')
        self.logger.info('Training instance with config: {}'.format(self.config))
        
        # model initialization
        self.model = hk.transform_with_state(self._forward_fn)
        self.rng_key = jax.random.PRNGKey(seed=42)
        self.params, self.st = self._load_init_model()

        # define optimizer
        self.optimizer = optax.adamw(learning_rate=self.learning_rate, b1=self.momentum, b2=0.999, eps=1e-8, weight_decay=self.weight_decay)

        # initialize optimizer state
        self.opt_state = self.optimizer.init(self.params)

        # define loss function
        self.value_loss_function = optax.l2_loss
        self.policy_loss_function = optax.softmax_cross_entropy

        # log losses for plotting
        self.total_loss_arr = []
        self.value_loss_arr = []
        self.policy_loss_arr = []
        self.iteration_arr = []

    def _load_init_model(self):
        dummy_x = jax.random.uniform(self.rng_key, (1, cw.BOARD_SIZE, cw.BOARD_SIZE, 2))
        params, st = self.model.init(rng=self.rng_key, x=dummy_x)
        return params, st

    # load model parameters, returns random initialized model if no model is found, else returns trained model of the latest iteration
    def _load_model(self):
        if self.self_play_iter > 0:
            self.logger.info('Loading trained model parameters from iteration: ' + str(self.self_play_iter - 1))
            params_path = os.path.join(self.trained_model_params_path, 'mod_{}'.format(self.self_play_iter - 1))
            opt_state_path = os.path.join(self.trained_model_params_path, 'opt_{}'.format(self.self_play_iter - 1))

            loaded_params = utils.load_model_params(params_path)
            self.params = loaded_params

            loaded_opt_state = utils.load_optimizer_state(opt_state_path)
            self.opt_state = loaded_opt_state

            # # assert that loaded params equal to params. (checking if params are loaded correctly)
            # if self.self_play_iter > 0:
            #     chex.assert_trees_all_equal(params, loaded_params)
            # self.params = params
        return
    
    # function to do forward pass on model
    def _forward_fn(self, x):
        model = AZeroModel.AlphaZeroModel(num_filters=256, training=True)
        return model(x)

    # filter out illegal moves
    def _remove_illegal(self, move_type, state, cc, reverse):
        get_moves = getattr(cc, 'get' + move_type)
        legal_moves = get_moves(state)
        legal_moves = pw.ll_to_list(legal_moves)
        legals = pw.moves_existence(legal_moves, reverse)
        return legals

    # define loss function
    def loss_fn(self, params, st, state, target_policy, target_outcome, mask, rng_key):
        forward, _st = self.model.apply(params=params, state=st, x=state, rng=rng_key)
        predict_outcome, predict_policy = forward

        # calculate value loss
        predict_outcome = predict_outcome.flatten()
        target_outcome = target_outcome.flatten()
        
        value_loss = jnp.mean(self.value_loss_function(predict_outcome, target_outcome))

        # mask illegal moves in prediction and calculate policy loss
        masked_predict_policy = jnp.where(target_policy, predict_policy, 0)
        policy_loss = jnp.mean(self.policy_loss_function(labels=target_policy, logits=masked_predict_policy))

        # # calculate total loss
        total_loss = value_loss + policy_loss

        return total_loss, (value_loss, policy_loss)

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, params, opt_state, st, state, policy, outcome, mask, rng_key):
        (loss, (value_loss, policy_loss)), grads = jax.value_and_grad(self.loss_fn, argnums=0, has_aux=True)(params, st, state, policy, outcome, mask, rng_key)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, value_loss, policy_loss

    def train(self, buffer_q, self_play_iter):
        start_time = time.time()

        print_list = ['epoch', 'step', 'total_loss', 'value_loss', 'policy_loss']
        self.logger.info(50 * "=")
        self.logger.info('{:^7} {:^7} {:^8} {:^8} {:^8}'.format(*print_list))
        self.logger.info(50 * "=")

        # load model parameters
        self.self_play_iter = self_play_iter
        self._load_model()

        # define dataset
        states, players, policy, outcomes = utils.create_dataset_from_buffer_q(buffer_q)

        # training loop
        for epoch in range(1):
            for i in range(0, states.shape[0], self.batch_size):
                # convert from strucutred array to jax array
                batch_states = states[i:i+self.batch_size]
                batch_policy = policy[i:i+self.batch_size]
                batch_outcome = outcomes[:, i:i+self.batch_size]

                # if batch size is not equal to batch_size, skip the batch. Happens at the end of the dataset
                if batch_states.shape[0] != self.batch_size:
                    break

                # convert vector to board
                p1 = pw.vec_to_board(batch_states, 1, self.batch_size)
                p2 = pw.vec_to_board(batch_states, 2, self.batch_size)

                # stack the two boards together
                batch_states = jnp.array(jnp.stack((p1, p2), axis=3)).astype(np.float32)
                
                # get legal moves
                legals_mask = []
                for j in range(self.batch_size):
                    s = batch_states[j]
                    state_str = ''.join([str(x) for x in s])
                    cc.applyState(state_str, gamestate)
                    legal_moves = self._remove_illegal('MovesForward', gamestate, cc, 0)
                    legals_mask.append(legal_moves)
                
                # convert mask to jax array
                legals_mask = jnp.array(legals_mask).astype(np.float32)
                
                self.params, self.opt_state, total_loss, value_loss, policy_loss = self.train_step(self.params, self.opt_state, self.st, batch_states, batch_policy, batch_outcome, legals_mask, self.rng_key)

                # save losses and iterations for plotting
                self.total_loss_arr.append(total_loss)
                self.value_loss_arr.append(value_loss)
                self.policy_loss_arr.append(policy_loss)
                self.iteration_arr.append(i//self.batch_size)

                # log losses and iterations in log file
                self.logger.info('{:^7} {:^7} {:^8.4f} {:^8.4f} {:^8.4f}'.format(epoch, i//self.batch_size, total_loss, value_loss, policy_loss))

            # save model
            # utils.save_model(self.trained_model_params_path+'mod_{}'.format(self.self_play_iter), self.params)
            # utils.save_optimizer_state(self.trained_model_params_path+'opt_{}'.format(self_play_iter), self.opt_state)

            # save total loss for plotting
            # utils.plot_loss(self.training_plots_path, self.self_play_iter, self.total_loss_arr, self.iteration_arr, name='total_loss')
            # utils.plot_loss(self.training_plots_path, self.self_play_iter, self.value_loss_arr, self.iteration_arr, name='value_loss')
            # utils.plot_loss(self.training_plots_path, self.self_play_iter, self.policy_loss_arr, self.iteration_arr, name='policy_loss')
            # utils.plot_losses_in_single_line(self.training_plots_path) # plot all losses in a single line

            # # plot losses and save to file
            # utils.plot(self.training_plots_path, self.iteration_arr, self.total_loss_arr, self.value_loss_arr, self.policy_loss_arr, self.self_play_iter)

            self.total_loss_arr = []
            self.value_loss_arr = []
            self.policy_loss_arr = []
            self.iteration_arr = []

        end_time = time.time()
        self.total_time = end_time - start_time
        self.logger.info('Training iteration: ' + str(self.self_play_iter) + ' completed')
        
        return self.total_time