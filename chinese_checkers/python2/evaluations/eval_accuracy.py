import os, sys
import numpy as np
import jax
import jax.numpy as jnp
import optax
import haiku as hk

# import file from outside directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from wrappers import pywrapper as pw
from wrappers import ccwrapper as cw
from models import AZeroModel
from helpers import utils

class Evaluation:
    def __init__(self):
        self.rng_key = jax.random.PRNGKey(42)
        self.dummy_x = jax.random.uniform(self.rng_key, (1, cw.BOARD_SIZE, cw.BOARD_SIZE, 2)) 
        self.model, self.params, self.st = self.init_model(self.rng_key)
        # self.test_dataset = utils.create_dataset_from_folder("/Users/bigyankarki/Desktop/bigyan/results/async_jax/data/")
        self.test_dataset = np.load('dataset.npy', allow_pickle=True)
        self.batch_size = self.test_dataset.shape[1]

    def init_model(self, rng):
        init_fn = hk.transform_with_state(self._forward_fn)
        input_shape = jax.random.normal(rng, (1, cw.BOARD_SIZE, cw.BOARD_SIZE, 2))
        params, st = init_fn.init(rng, input_shape)
        return init_fn, params, st
    
    def _forward_fn(self, x):
        model = AZeroModel.AlphaZeroModel(num_filters=256, training=True)
        return model(x)
    
    def evaluate(self, model_path):
        self.params = utils.load_model_params(model_path)

        state, y_target_policy, y_target_value = self.test_dataset

        state = jnp.array([state[i] for i in range(len(state))], dtype=jnp.float32)
        y_target_policy = jnp.array([y_target_policy[i] for i in range(len(y_target_policy))], dtype=jnp.float32)
        y_target_value = jnp.array([y_target_value[i] for i in range(len(y_target_value))], dtype=jnp.float32)
    
        batch_state = state[:self.batch_size]

        # convert vector to board
        p1 = pw.vec_to_board(batch_state, 1, self.batch_size)
        p2 = pw.vec_to_board(batch_state, 2, self.batch_size)

        # stack the two boards together
        batch_state = jnp.array(jnp.stack((p1, p2), axis=3)).astype(np.float32)

        # evaluate modell
        (y_predict_value, y_predict_policy), _st = self.model.apply(params=self.params, state=self.st, x=batch_state, rng=self.rng_key)

        # calculate value loss
        value_loss = jnp.mean(optax.l2_loss(y_predict_value.flatten(), y_target_value.flatten()))

        # mask out invalid moves and calculate policy loss
        y_predict_policy = jnp.where(jnp.array(y_target_policy), y_predict_policy, 0)
        policy_loss = jnp.mean(optax.softmax_cross_entropy(labels=y_target_policy, logits=y_predict_policy))

        # calculate accuracy of value 
        y_predict_value = y_predict_value.flatten() # flatten y_predict_label
        y_predict_value = jnp.where(y_predict_value <= 0, -1, 1)  # set y_predict_label to -1 if value is <=0  or else 1
        y_target_value = y_target_value.flatten() # flatten y_label

        # calculate accuracy
        accuracy = jnp.mean(jnp.where(y_target_value == y_predict_value, 1, 0))

        return accuracy, value_loss, policy_loss
    

if __name__ == '__main__':
    evaluation = Evaluation()

    # evaluate model
    # accuracy, value_loss, policy_loss = evaluation.evaluate("/Users/bigyankarki/Desktop/bigyan/results/latest_models/hour_71/parallel_play_model_124.pkl")
    accuracy, value_loss, policy_loss = evaluation.evaluate("/Users/bigyankarki/Desktop/bigyan/results/19_res_blocks/mod_166")

    print('Accuracy: {}, Value Loss: {}, Policy Loss: {}'.format(accuracy, value_loss, policy_loss))







       

# # function to initialize Haiku model with rng key and input shape
# def init_model(rng):
#     init_fn = hk.transform_with_state(_forward_fn)
#     input_shape = jax.random.normal(rng, (1, cw.BOARD_SIZE, cw.BOARD_SIZE, 2))
#     params, st = init_fn.init(rng, input_shape)
#     return init_fn, params, st

# function to do forward pass on model
# def _forward_fn(x):
#     model = AZeroModel.AlphaZeroModel(num_filters=256, training=True)
#     return model(x)
    

# function to evaluate model on different dataset
# def evaluate(model, params, st, rng):
#     # load test dataset
#     test_dataset = np.load('test_set.npy', allow_pickle=True)
#     # print('Loaded test dataset with {} samples'.format(test_dataset.shape[1]))

#     # batch the dataset
#     batch_size = test_dataset.shape[1] # a single batch

#     # convert from strucutred array to jax array
#     state = test_dataset[0]
#     y_policy = test_dataset[1]
#     y_value = test_dataset[2]
#     state = jnp.array([state[i] for i in range(len(state))], dtype=jnp.float32)
#     y_target_policy = jnp.array([y_policy[i] for i in range(len(y_policy))], dtype=jnp.float32)
#     y_target_value = jnp.array([y_value[i] for i in range(len(y_value))], dtype=jnp.float32)

#     batch_state = state[:batch_size]
#     # print(batch_state.shape)

#     # # convert vector to board
#     p1 = pw.vec_to_board(batch_state, 1, batch_size)
#     p2 = pw.vec_to_board(batch_state, 2, batch_size)

#     # stack the two boards together
#     batch_state = jnp.array(jnp.stack((p1, p2), axis=3)).astype(np.float32)

#     # evaluate modell
#     (y_predict_value, y_predict_policy), _st = model.apply(params=params, state=st, x=batch_state, rng=rng)

#     # calculate value loss
#     value_loss = jnp.mean(optax.l2_loss(y_predict_value.flatten(), y_target_value.flatten()))

#     # mask out invalid moves and calculate policy loss
#     y_predict_policy = jnp.where(jnp.array(y_target_policy), y_predict_policy, 0)
#     policy_loss = jnp.mean(optax.softmax_cross_entropy(labels=y_target_policy, logits=y_predict_policy))

#     # calculate accuracy of value 
#     y_predict_value = y_predict_value.flatten() # flatten y_predict_label
#     y_predict_value = jnp.where(y_predict_value <= 0, -1, 1)  # set y_predict_label to -1 if value is <=0  or else 1
#     y_value = y_value.flatten() # flatten y_label

#     # calculate accuracy
#     accuracy = jnp.mean(jnp.where(y_target_value == y_predict_value, 1, 0))

#     return accuracy, value_loss, policy_loss

# def evaluation_from_path(model_path):
#     # model initialization
#     model = hk.transform_with_state(_forward_fn)
#     rng_key = jax.random.PRNGKey(42)
#     dummy_x = jax.random.uniform(rng_key, (1, cw.BOARD_SIZE, cw.BOARD_SIZE, 2))
#     params, st = model.init(rng=rng_key, x=dummy_x)

#     # get params from saved model
#     loaded_params = utils.load_model_params(model_path)
#     params = loaded_params

#     # assert params are equal to loaded params and not init params
#     chex.assert_trees_all_equal(params, loaded_params)
#     # evaluate model
#     accuracy, value_loss, policy_loss = evaluate(model, params, st, rng_key)
#     return accuracy, value_loss, policy_loss



# if __name__ == '__main__':
#     mods = [0, 1, 5, 10]

#     # model initialization
#     model = hk.transform_with_state(_forward_fn)
#     rng_key = jax.random.PRNGKey(seed)
#     dummy_x = jax.random.uniform(rng_key, (1, cw.BOARD_SIZE, cw.BOARD_SIZE, 2))
#     params, st = model.init(rng=rng_key, x=dummy_x)

#     print_list = ['mod', 'Accuracy', 'value_loss', 'policy_loss']
#     logger.info('{:^8} {:^8} {:^8} {:^8}'.format(*print_list))

#     for mod in mods:
#         # get params from saved model
#         path = os.path.join(trained_model_params_path, 'mod_{}'.format(mod - 1))
#         loaded_params = utils.load_model_params(path)
#         params = loaded_params
    
#         # assert params are equal to loaded params and not init params
#         chex.assert_trees_all_equal(params, loaded_params)

#         # evaluate model
#         accuracy, value_loss, policy_loss = evaluate(model, params, st, rng_key)
#         logger.info('{:^8} {:^8.4f} {:^8.4f} {:^8.4f}'.format(mod, accuracy, value_loss, policy_loss))


