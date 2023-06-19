# AlphaZero model in Haiku
import haiku as hk
import jax
import jax.numpy as jnp
import os
# from wrappers import ccwrapper as cw

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

# class for the Residual Block (ResNet)
class ResidualBlock(hk.Module):
    def __init__(self, num_filters, training, name=None):
        super().__init__(name=name)
        self.num_filters = num_filters
        self.training = training

    def __call__(self, x):
        conv1 = hk.Conv2D(output_channels=self.num_filters, kernel_shape=3, stride=1, padding='SAME', with_bias=False)
        conv2 = hk.Conv2D(output_channels=self.num_filters, kernel_shape=3, stride=1, padding='SAME', with_bias=False)
        bn1 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9, eps=1e-5)
        bn2 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9, eps=1e-5)
        relu = jax.nn.relu
        shortcut = x
        
        x = conv1(x)
        x = bn1(x, is_training=self.training)
        x = relu(x)
        x = conv2(x)
        x = bn2(x, is_training=self.training)
        x = x + shortcut
        x = relu(x)
        return x

# class for the Policy Head
class PolicyHead(hk.Module):
    def __init__(self, num_filters, training, name=None):
        super().__init__(name=name)
        self.num_filters = num_filters
        self.training = training

    def __call__(self, x):
        conv = hk.Conv2D(output_channels=self.num_filters, kernel_shape=1, stride=1, padding='SAME', with_bias=False)
        bn = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9, eps=1e-5)
        flatten = hk.Flatten()
        relu = jax.nn.relu

        x = conv(x)
        x = bn(x, is_training=self.training)
        x = relu(x)
        x = flatten(x)
        x = hk.Linear(5**4)(x)
        return x

# class for the Value Head
class ValueHead(hk.Module):
    def __init__(self, num_filters, training, name=None):
        super().__init__(name=name)
        self.num_filters = num_filters
        self.training = training

    def __call__(self, x):
        flatten = hk.Flatten()
        relu = jax.nn.relu

        x = flatten(x)
        x = hk.Linear(256)(x)
        x = relu(x)
        x = hk.Linear(1)(x)
        x = jnp.tanh(x)
        return x

# class for the AlphaZero model
class AlphaZeroModel(hk.Module):
    def __init__(self, num_filters, training, name=None):
        super().__init__(name=name)
        self.num_filters = num_filters
        self.training = training
    
    def __call__(self, x):
        conv = hk.Conv2D(output_channels=self.num_filters, kernel_shape=3, stride=1, padding='SAME', with_bias=False)
        bn = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9, eps=1e-5)
        relu = jax.nn.relu

        x = conv(x)
        x = bn(x, is_training=self.training)
        x = relu(x)
        for _ in range(18):
            x = ResidualBlock(self.num_filters, training=self.training)(x)

        policy = PolicyHead(self.num_filters, training=self.training)(x)
        value = ValueHead(self.num_filters, training=self.training)(x)
        return value, policy


if __name__ == "__main__":
    # function to do forward pass on model
    def _forward_fn(x):
        model = AlphaZeroModel(256, training=True)
        return model(x)

    # function to initialize Haiku model with rng key and input shape
    def init_model(rng):
        transformed_func = hk.transform_with_state(_forward_fn)
        input_shape = jax.random.normal(rng, (1, 5, 5, 1))
        params, st = transformed_func.init(rng, input_shape)
        return transformed_func, params, st

    # @partial(jax.jit, static_argnums=(0,))
    def forward_pass(model, params, st, x, rng):
        return model.apply(params=params, state=st, x=x, rng=rng)

    if __name__ == '__main__':
        rng = jax.random.PRNGKey(42)
        model, params, st = init_model(rng)
        
        # do a forward pass
        x = jax.random.normal(rng, (1, 5, 5, 1))
        for i in range(10):
            v, p = forward_pass(model, params=params, st=st, x=x, rng=rng)
            # print
            (v, p), st = forward_pass(model, params=params, st=st, x=x, rng=rng)
            print(v.shape, p.shape)

        

