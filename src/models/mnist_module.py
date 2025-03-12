from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
import optax
import reax
import torch
from typing_extensions import override


class MnistModule(reax.Module):
    """Example of a `reax.Module` for MNIST classification.

    A `reax.Module` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Callable[[...], optax.GradientTransformation],
        scheduler: Optional[optax.Schedule],
        compile: bool,
    ) -> None:
        """Initialize a `MnistModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False)

        self.net = net
        self._optimizer: Callable[[...], optax.GradientTransformation] = optimizer
        self._scheduler: Optional[optax.Schedule] = scheduler
        if compile:
            self.model_step = jax.jit(self.model_step, static_argnums=[4, 5, 6])
        self._batch_stats: Optional[dict] = None

        self.loss_fn = optax.losses.safe_softmax_cross_entropy

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = reax.metrics.Accuracy(mode="multiclass", num_classes=10)
        self.val_acc = reax.metrics.Accuracy(mode="multiclass", num_classes=10)
        self.test_acc = reax.metrics.Accuracy(mode="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = reax.metrics.Average()
        self.val_loss = reax.metrics.Average()
        self.test_loss = reax.metrics.Average()

        # for tracking best so far validation accuracy
        self.val_acc_best = reax.metrics.Max()

    @property
    def pytree(self) -> dict:
        return {"params": self.parameters(), "batch_stats": self._batch_stats}

    def forward(self, x: jax.Array, train: bool) -> jax.Array:
        """Perform a forward pass through the model `self.net`.

        :param x: An array of images.
        :return: An array of logits.
        """
        return self._forward(self.net, self.pytree, train, x)

    @staticmethod
    def _forward(model, pytree, train, *args, **kwargs):
        return model.apply(
            pytree,
            *args,
            **kwargs,
            mutable=["batch_stats"] if train else False,
            train=train,
        )

    @staticmethod
    def model_step(
        params: dict,
        batch_stats: dict,
        x: jax.Array,
        y: jax.Array,
        model: Callable[[jax.Array], jax.Array],
        loss_fn,
        train: bool = True,
    ) -> tuple[jax.Array, tuple[jax.Array, dict]]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        pytree = {"params": params, "batch_stats": batch_stats}
        outs = MnistModule._forward(model, pytree, train, x)
        logits, new_batch_stats = outs if train else (outs, None)
        loss = loss_fn(logits, y.astype(jnp.float32)).mean()
        aux_data = logits, new_batch_stats
        return loss, aux_data

    @override
    def training_step(
        self, batch: tuple[jax.Array, jax.Array], batch_idx: int
    ) -> tuple[jax.Array, jax.Array]:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        x, targets = batch
        val_with_grad = jax.value_and_grad(self.model_step, has_aux=True)
        (loss, aux_data), grads = val_with_grad(
            self.parameters(),
            self._batch_stats,
            x,
            targets,
            self.net,
            self.loss_fn,
            train=True,
        )
        logits, new_model_state = aux_data

        # Update the batch stats
        self._batch_stats = new_model_state["batch_stats"]

        # update and log metrics
        self.log(
            "train/loss",
            self.train_loss.create(loss),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/acc",
            self.train_acc.create(logits, jnp.argmax(targets, axis=1, keepdims=True)),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # return loss or backpropagation will fail
        return loss, grads

    @override
    def validation_step(self, batch: tuple[jax.Array, jax.Array], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        x, targets = batch
        loss, (logits, _) = self.model_step(
            self.parameters(),
            self._batch_stats,
            x,
            targets,
            self.net,
            self.loss_fn,
            train=False,
        )

        # update and log metrics
        self.log(
            "val/loss",
            self.val_loss.create(loss),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/acc",
            self.train_acc.create(logits, jnp.argmax(targets, axis=1, keepdims=True)),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch: tuple[jax.Array, jax.Array], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        x, targets = batch
        loss, (logits, _) = self.model_step(
            self.parameters(),
            self._batch_stats,
            x,
            targets,
            self.net,
            self.loss_fn,
            train=False,
        )

        # update and log metrics
        self.log(
            "test/loss",
            self.test_loss.create(loss),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/acc",
            self.train_acc.create(logits, jnp.argmax(targets, axis=1, keepdims=True)),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_model(self, stage: str, batch, /) -> None:
        """REAX hook that is called at the beginning of fit (train + validate), validate, test, or
        predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.parameters() is None:
            images, _labels = batch
            state = self.net.init(self.rng_key(), images)
            params = state["params"]
            self.set_parameters(params)
            self._batch_stats = state["batch_stats"]

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        :return: A dict containing the configured optimizers and learning-rate schedulers to be
            used for training.
        """
        if self._scheduler is None:
            opt = self._optimizer()
        else:
            # Assume the scheduler can be used as a learning rate function
            opt = self._optimizer(learning_rate=self._scheduler)

        state = opt.init(self.parameters())
        return opt, state

    @override
    def state_dict(self) -> dict[str, Any]:
        ckpt = super().state_dict()
        ckpt["batch_stats"] = self._batch_stats
        return ckpt

    @override
    def load_state(self, state_dict: dict[str, Any]) -> None:
        super().load_state(state_dict)
        self._batch_stats = state_dict["batch_stats"]


if __name__ == "__main__":
    _ = MnistModule(None, None, None, None)
