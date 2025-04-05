from flax import linen
import jax
import jax.numpy as jnp


class SimpleDenseNet(linen.Module):
    """A simple fully-connected neural net for computing predictions.

    :param input_size: The number of input features.
    :param lin1_size: The number of output features of the first linear layer.
    :param lin2_size: The number of output features of the second linear layer.
    :param lin3_size: The number of output features of the third linear layer.
    :param output_size: The number of output features of the final linear layer.
    """

    input_size: int = 784
    lin1_size: int = 256
    lin2_size: int = 256
    lin3_size: int = 256
    output_size: int = 10

    @linen.compact
    def __call__(self, x: jax.Array, train=True) -> jax.Array:
        """Perform a single forward pass through the network.

        :param x: The input array.
        :return: An array of predictions.
        """
        x = x.reshape((x.shape[0], -1))

        x = linen.Dense(self.lin1_size)(x)
        x = linen.BatchNorm()(x, use_running_average=not train)
        x = linen.relu(x)
        x = linen.Dense(self.lin2_size)(x)
        x = linen.BatchNorm()(x, use_running_average=not train)
        x = linen.relu(x)
        x = linen.Dense(self.lin3_size)(x)
        x = linen.BatchNorm()(x, use_running_average=not train)
        x = linen.relu(x)
        x = linen.Dense(self.output_size)(x)

        return x


if __name__ == "__main__":
    _ = SimpleDenseNet()
