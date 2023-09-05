"Linear modules." ""

from typing import (
    Any,
    Callable,
    Optional,
    Tuple,
    Union,
)

from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen.module import compact
from flax.linen.module import Module
from jax import lax
import jax.numpy as jnp
import jax


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[
    None,
    str,
    lax.Precision,
    Tuple[str, str],
    Tuple[lax.Precision, lax.Precision],
]
DotGeneralT = Callable[..., Array]
ConvGeneralDilatedT = Callable[..., Array]

default_kernel_init = initializers.lecun_normal()


class MultiHeadDense(Module):
    """A linear transformation applied over the last dimension of the input.

    Attributes:
      features: the number of output features.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
    """

    features: int
    num_heads: int
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    # Deprecated. Will be removed.
    dot_general: DotGeneralT = lax.dot_general
    dot_general_cls: Any = None

    @compact
    def __call__(self, inputs: Array, indices: Array) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.num_heads, jnp.shape(inputs)[-1], self.features),
            # (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )
        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.num_heads, self.features,), self.param_dtype
                # "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        if self.dot_general_cls is not None:
            dot_general = self.dot_general_cls()
        else:
            dot_general = self.dot_general

        def apply_dot_general(idx):
            __y = dot_general(
                inputs[idx][None, ...],
                kernel[idx],
                (((inputs.ndim - 1,), (0,)), ((), ())),
                precision=self.precision,
            )
            if bias is not None:
                __y += jnp.reshape(bias[idx], (1,) * (__y.ndim - 1) + (-1,))
            return __y

        y = jax.vmap(apply_dot_general)(indices)
        y = jnp.concatenate(y)
        return y

def main():
    key = jax.random.PRNGKey(0)
    N = 128
    mh_dense = MultiHeadDense(features=100, num_heads=10)
    obs = jnp.zeros((N, 100))
    task_id = jax.random.randint(key, shape=(N,),minval=0, maxval=10)
    y, variables = mh_dense.init_with_output(key, obs, task_id)
    print(y.shape)

if __name__ == '__main__':
    main()
