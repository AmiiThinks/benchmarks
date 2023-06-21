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
        self.test_dataset = None
        self.batch_size = 1
        # self.params = utils.load_model_params(params)

    def init_model(self, rng):
        init_fn = hk.transform_with_state(self._forward_fn)
        input_shape = jax.random.normal(rng, (1, cw.BOARD_SIZE, cw.BOARD_SIZE, 2))
        params, st = init_fn.init(rng, input_shape)
        return init_fn, params, st
    
    def _forward_fn(self, x):
        model = AZeroModel.AlphaZeroModel(num_filters=256, training=True)
        return model(x)
    
    def evaluate(self, test_dataset, params):
        self.params = utils.load_model_params(params)
        self.test_dataset = test_dataset
        self.batch_size = len(self.test_dataset)

        state = self.test_dataset

        state = jnp.array([state[i] for i in range(len(state))], dtype=jnp.float32)
    
        batch_state = state[:self.batch_size]

        # convert vector to board
        p1 = pw.vec_to_board(batch_state, 1, self.batch_size)
        p2 = pw.vec_to_board(batch_state, 2, self.batch_size)

        # stack the two boards together
        batch_state = jnp.array(jnp.stack((p1, p2), axis=3)).astype(np.float32)

        # evaluate modell
        (y_predict_value, y_predict_policy), _st = self.model.apply(params=self.params, state=self.st, x=batch_state, rng=self.rng_key)

        # calculate accuracy of value 
        y_predict_value = y_predict_value.flatten() # flatten y_predict_label
        y_predict_value = jnp.where(y_predict_value <= 0, -1, 1)  # set y_predict_label to -1 if value is <=0  or else 1

        return y_predict_value, y_predict_policy
    

if __name__ == '__main__':
    evaluation = Evaluation()

    # evaluate model
    # accuracy, value_loss, policy_loss = evaluation.evaluate("/Users/bigyankarki/Desktop/bigyan/results/latest_models/hour_71/parallel_play_model_124.pkl")
    accuracy, value_loss, policy_loss = evaluation.evaluate("/Users/bigyankarki/Desktop/bigyan/results/19_res_blocks/mod_166")

    print('Accuracy: {}, Value Loss: {}, Policy Loss: {}'.format(accuracy, value_loss, policy_loss))
