from datetime import datetime
from helper import ReshapeTransform
from functools import partial
from collections import namedtuple
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F
from torch import nn
import torch
from mpl2latex import mpl2latex, latex_figsize
#from sklearn.datasets import fetch_openml
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, List
#from scipy.integrate import solve_ivp
#import os


# Load (normalized) MNIST dataset, split into train/validation/test, reshape to 1D vectors.

# From: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir: str = '../../data/project/',
                 train_batch_size: int = 100,
                 test_batch_size: int = 1000):

        super().__init__()

        #---Store arguments---#
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        #---Transforms---#
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            ReshapeTransform((-1,))
            #transforms.Lambda(lambda x: x.double()),
        ])

        self.dims = (28**2,)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            print('yes')
            mnist_full = MNIST(self.data_dir, train=True,
                               transform=self.transforms, download=True)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [50000, 10000])  # Train/Validation split

        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.train_batch_size, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.test_batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.test_batch_size, pin_memory=True)


class Local(Optimizer):
    """
    A simple optimizer that changes weights according to `weights += learning_rate * delta_weights`,
    for a provided `delta_weights`, automatically retrieving the `learning_rate` from each layer metadata.

    Taken from: https://github.com/Joxis/pytorch-hebbian/blob/595ec79577c3816c61d39b0633f8dbf14d28b67a/pytorch_hebbian/optimizers/local.py#L15
    """

    def __init__(self, named_params, lr=required):
        self.param_names, params = zip(*named_params)

        # torch.optim.optimizer.required checks wheter a particular gradient is required for a parameter
        # during the optimization step.
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(Local, self).__init__(params, defaults)

    def local_step(self, delta_weights, layer_name, closure=None):
        """Performs a single local optimization step: weights += learning_rate * delta_weights.
        The needed learning_rate is automatically retrieved from the layer metadata."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            layer_index = self.param_names.index(
                layer_name + '.weight')  # find index of layer
            weights = group['params'][layer_index]
            #print('lr', group['lr'])
            weights.data.add_(group['lr'] * delta_weights)
            #print("Add weights to ", layer_index)

        try:
            self._step_count += 1
        except AttributeError:
            pass

        return loss


class BioLearningRule():
    """
    Implements the bio-inspired Hebbian learning rule from [1]. 

    Adapted from https://github.com/Joxis/pytorch-hebbian/blob/master/pytorch_hebbian/learning_rules/krotov.py

    [1]: "Unsupervised learning by competing hidden units", D. Krotov, J. J. Hopfield, 2019, 
         https://www.pnas.org/content/116/16/7723
    """

    def __init__(self,
                 precision: float = 1e-30,
                 delta: float = 0.4,
                 lebesgue_p: float = 2.0,
                 ranking_param: int = 2):
        """
        Parameters
        ----------
        precision : float
            Minimum value that is considered non-zero.
        delta : float
            Strength of anti-Hebbian learning (from eq. 9 in [1]). 
        lebesgue_p : float
            Parameter for Lebesgue measure, used for defining an inner product (from eq. 2 in [1]).
        ranking_param: int
            Rank of the current to which anti-hebbian learning is applied. Should be >= 2. 
            This is the `k` from eq. 10 in [1].
        """

        # Store hyperparameters
        self.precision = precision
        self.delta = delta
        self.lebesgue_p = lebesgue_p
        self.ranking_param = ranking_param

        print(self)

    def __str__(self):
        """Return string representation of the object"""

        return "Krotov-Hopfield Hebbian learning rule (delta={:.2f}, lebesgue_p={:.2f}, ranking_param={})".format(self.delta, self.lebesgue_p, self.ranking_param)

    def __repr__(self):
        """Return string representation of the object"""

        return str(self)

    def __call__(self,
                 inputs: torch.Tensor,
                 weights: torch.Tensor) -> torch.tensor:
        """
        Compute the change of `weights` given by the Krotov learning rule (eq. 3 from [1], with R=1)
        for a given batch of `inputs`. See "A fast implementation" section in [1].

        The formula is:
        ```delta_weights = g(currents) @ inputs - normalization_mtx (*) weights
        currents = (sgn(weights) (*) abs(weights) ** lebesgue_p) @ inputs.T```

        where `normalization_mtx` is a matrix of the same shape of `weights`, with all columns equal to:
        ```\sum_{batches} [g(currents) (*) currents]```

        The symbol `@` denotes matrix multiplication, while `(*)` is element-wise multiplication (Hadamard product).
        Finally, the function `g` (eq. 10 in [1]) returns:
        ```g(currents[i,j]) = 1      if currents[i,j] is the highest in the j-th column (sample),
                              -Delta if currets[i,j] is the k-th highest value in the j-th column (sample),
                               0     otherwise```


        Parameters
        ----------
        inputs : torch.Tensor of shape (batch_size, input_size)
            Batch of inputs
        weights : torch.Tensor of shape (output_size, input_size)
            Model's weights

        Returns
        -------
        delta_weights : torch.Tensor of shape (output_size, input_size)
            Change of weights given by the fast implementation of Krotov learning rule. The tensor is normalized
            so that its maximum is equal to 1.
        """

        batch_size = inputs.shape[0]

        #---Currents---#
        # Shape is (batch_size, input_size) -> (input_size, batch_size)
        inputs = torch.t(inputs)
        currents = torch.matmul(torch.sign(weights) * torch.abs(weights) ** (
            self.lebesgue_p - 1), inputs)  # Shape is (output_size, batch_size)

        #---Activations---#
        # for the documentation of topk see https://pytorch.org/docs/stable/generated/torch.topk.html
        # Shape is (self.ranking_param, batch_size)
        _, ranking_indices = currents.topk(self.ranking_param, dim=0)
        # Indices of the top k currents produced by each input sample

        post_activation_currents = torch.zeros_like(
            currents)  # Shape is (output_size, batch_size)
        # Computes g(currents)
        # Note that all activations are 0, except the largest current (activation of 1) and the k-th largest (activation of -delta)

        batch_indices = torch.arange(
            batch_size, device=post_activation_currents.device)
        post_activation_currents[ranking_indices[0], batch_indices] = 1.0
        post_activation_currents[ranking_indices[self.ranking_param-1],
                                 batch_indices] = -self.delta

        #---Compute change of weights---#
        # Overlap between post_activation_currents and inputs
        delta_weights = torch.matmul(post_activation_currents, torch.t(inputs))
        # Overlap between currents and post_activation_currents
        second_term = torch.sum(
            torch.mul(post_activation_currents, currents), dim=1)
        # Results are summed over batches, resulting in a shape of (output_size,)

        delta_weights = delta_weights - second_term.unsqueeze(1) * weights

        #---Normalize---#
        nc = torch.max(torch.abs(delta_weights))
        if nc < self.precision:
            nc = self.precision

        # Maximum (absolute) change of weight is set to +1.
        return delta_weights / nc


class Net(nn.Module):

    def __init__(self,
                 input_dim: int = 28*28,
                 output_dim: int = 10,
                 num_hidden: int = 2000,
                 n: int = 4.5,
                 beta: float = .1):
        """Builds the standard feed-forward neural network architecture presented in [1].

        Parameters
        ----------
        input_dim : int
            Input dimensionality
        output_dim : int
            Output dimensionality (number of classes)
        num_hidden : int
            Number of hidden units
        n : int
            Exponent of activation function for the first hidden layer (n = 1 for ReLU). See eq. 1 in [1].
        beta : float
            Parameter for the activation function in the top layer (akin to an "inverse temperature"). See eq. 1 in [1].         

        [1]: "Unsupervised learning by competing hidden units", D. Krotov, J. Hopfield, 2019
        """

        super(Net, self).__init__()

        # Store parameters
        self.n = n
        self.beta = beta

        # Define layers
        self.num_hidden = num_hidden
        self.hidden = nn.Linear(input_dim, self.num_hidden, bias=False)
        self.top = nn.Linear(self.num_hidden, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the output of the network for the input `x`.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, self.input_dim)
            Batch of input values

        Returns
        -------
        out : torch.Tensor of shape (batch_size, self.output_dim)
            Batch of output values
        """

        x = F.relu(self.hidden(x))**self.n
        x = torch.tanh(self.beta * self.top(x))

        return x


class UnsupervisedPhase(pl.LightningModule):
    """
    Perform unsupervised learning of the first layers of a sequential network according to some local learning rule.
    """

    def __init__(self,
                 model: torch.nn.Sequential,
                 update_weights_rule: Callable,
                 supervised_from: int = -1,
                 freeze_layers: List[str] = None,
                 n_epochs: int = 100,
                 lr: float = 0.02
                 ):
        """
        Parameters
        ----------
        model : torch.nn.Sequential
            Model to be trained. Must be a sequence of Linear layers (+ some activations).
        update_weights_rule : function
            Function that computes the change of weights
        supervised_from : int
            Train all layers that appear *before* `supervised_from`. By default, all layers except the very last one are trained.
        freeze_layers : list of strings
            Names of layers to be excluded from training.
        n_epochs : int
            Number of epochs for the full training (needed to set the learning rate)
        lr : float
            Initial Learning Rate
        """

        super().__init__()

        #---Store arguments---#
        self.model = model
        self.supervised_from = supervised_from
        self.freeze_layers = freeze_layers
        self.update_weights_rule = update_weights_rule
        self.n_epochs = n_epochs
        self.lr = lr

        if self.freeze_layers is None:
            self.freeze_layers = []

        self.layers = []  # Save all layers that need to be trained
        # simple object to store layer metadata
        Layer = namedtuple('Layer', ['idx', 'name', 'layer'])

        for idx, (name, layer) in enumerate(list(model.named_children())[:self.supervised_from]):
            if (type(layer) == torch.nn.Linear) and name not in self.freeze_layers:
                self.layers.append(Layer(idx, name, layer))

                # and initialize weights with normal distribution
                layer.weight.data.normal_(mean=0.0, std=1.0)

        print("Layers selected for unsupervised local (bio)learning:")
        print("{} layer(s): ".format(len(self.layers)),
              [lyr.name for lyr in self.layers])

        # Register hooks in trainable layers so that input/output can be stored (which is needed for Hebbian learning)
        self._hooks = {}
        self._inputs = {}
        self._outputs = {}

        for layer in self.layers:
            self._hooks[layer.name] = layer.layer.register_forward_hook(
                partial(self._store_data_hook, layer_name=layer.name)
            )

        self.automatic_optimization = False  # Disable automatic gradient optimizations
        # (since we are using a local optimizer, which does not need any gradient)

    def forward(self, x: torch.Tensor):
        """Compute model output"""

        return self.model.forward(x)

    def _store_data_hook(self, _, inp, out, layer_name):
        """Hook for storing input/outputs of trainable layers"""

        #print("Hook called with name: ", layer_name)
        self._inputs[layer_name] = inp[0]
        self._outputs[layer_name] = out

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Single iteration of the training cycle.
        """

        opt = self.optimizers()  # Retrieve the optimizer

        with torch.no_grad():  # No gradient is needed
            x, y = batch

            #---Forward step---#
            layers = list(self.model.children())
            for layer in layers[:self.supervised_from]:
                x = layer(x)

            #---Unsupervised learning---#
            for layer_idx, layer_name, layer in self.layers:
                inputs = self._inputs[layer_name]

                weights = layer.weight

                # Compute change of weights
                delta_weights = self.update_weights_rule(inputs, weights)
                delta_weights = delta_weights.view(
                    *layer.weight.size())  # Flatten

                # Update weights
                opt.local_step(delta_weights, layer_name=layer_name)

        return x, y

    def configure_optimizers(self):
        """Define optimizer and learning rate schedule."""

        start_lr = self.lr
        optimizer = Local(
            named_params=self.model.named_parameters(), lr=start_lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (self.n_epochs - epoch) / self.n_epochs,
                                                         verbose=True)

        return [optimizer], [lr_scheduler]


class VisualizationCallbacks(pl.Callback):
    """
    Interface with the comet.ml logger to visualize weights during training, and store all hyperparameters of a run.
    """

    def __init__(self, step_frequency: int = 250):
        # Visualize weights every `self.step_frequency` training steps.
        self.step_frequency = step_frequency

    """
    @staticmethod
    def draw_weights(weights: np.ndarray,
                     reshape_dim: Tuple[int, int] = (28, 28),
                     max_per_row: int = 5,
                     max_rows: int = 5,
                     epoch: int = -1):
        
    Plot the first few weights as matrices. `weights` should be an array of shape(output_dim, input_dim), i.e.
    `weights[i, j]` is the weight connecting the $j$- th neuron of a layer $n$ to the $i$- th neuron of the $n+1$ layer.
    Namely, all the weights connected to the $i$- th output neuron are the ones in the $i$- th row of `weights`.
    These weights are reshaped according to `reshape_dim` to construct a matrix. The weight matrices of the first neurons
    are then plotted in a grid of up to `max_rows` rows and `max_per_row` columns.
    

        # Shape of weights is (output_dim, input_dim)
        fig = plt.figure()
        nc = np.max(np.abs(weights))  # (Absolute) range of weights

        n_neurons = weights.shape[0]

        #---Infer number of rows/columns---#
        n_columns = max_per_row
        n_rows = n_neurons // max_per_row

        if n_rows > max_rows:
            n_rows = max_rows
        if n_rows == 1:
            n_columns = n_neurons
        if n_neurons > max_rows * max_per_row:
            n_neurons = max_rows * max_per_row

        #---Generate grid---#
        whole_image = np.zeros(reshape_dim * np.array([n_rows, n_columns]))

        i_row = 0
        i_col = 0
        size_x, size_y = reshape_dim

        plt.clf()  # Clear figure
        plt.tight_layout()

        for index_neuron in range(n_neurons):
            img = weights[index_neuron, ...].reshape(reshape_dim)
            whole_image[i_row * size_x:(i_row+1) * size_x,
                        i_col * size_y:(i_col+1) * size_y] = img
            i_col += 1

            if (i_col >= n_columns):
                i_col = 0
                i_row += 1

        #---Plot---#
        img_plotted = plt.imshow(
            whole_image, cmap='bwr', vmin=-nc, vmax=nc, interpolation=None)
        fig.colorbar(img_plotted, ticks=[np.amin(
            whole_image), 0, np.amax(whole_image)], ax=plt.gca())

        plt.axis('off')
        plt.show()

        # Save images for animation
        fig.savefig(
            f"../../data/project/weightsUnsupervised/weights{epoch}.png", transparent=True, bbox_inches='tight')

        return fig
    """

    def on_batch_end(self, trainer, pl_module):
        """
        Draw weights every `self.step_frequency` iterations.
        """

        epoch = trainer.current_epoch
        global_step = trainer.global_step

        """
        if global_step % self.step_frequency == 0:
            weights = pl_module.model.hidden.weight.detach().cpu().numpy()
            norms = np.sum(np.abs(weights) **
                           (pl_module.update_weights_rule.lebesgue_p), axis=1)
            print('Normalizations: ', np.mean(norms), ' +- ', np.std(norms))
            fig = self.draw_weights(weights, epoch=epoch)
            pl_module.logger.experiment.log_figure(
                figure_name="weights", figure=fig, step=global_step)
            plt.close()
        """

    def on_train_start(self, trainer, pl_module):
        """
        Store the hyperparameters at the start of training.
        """

        pl_module.logger.experiment.log_parameters(
            {'delta': pl_module.update_weights_rule.delta,
             'ranking_param': pl_module.update_weights_rule.ranking_param,
             'lebesgue_p': pl_module.update_weights_rule.lebesgue_p,
             'n': pl_module.model.n,
             'beta': pl_module.model.beta,
             'arch': str(pl_module.model)
             }
        )


#---Logger---#
# Logs hyperparameters and visualizations on an online server. Requires a free comet.ml account (https://www.comet.ml/).
# The personal api_key should be inserted below.
# The model can run also without the logger: it suffices to remove "callbacks=[visualization]" from the trainer definition.

# comet_logger = CometLogger(
#     api_key="<my_api_key>",
#     project_name="my_project",
#     workspace="my_workspace",
#     experiment_name="unsupervised_biolearning"
# )
visualization = VisualizationCallbacks(step_frequency=500)

# trainer = Trainer(gpus=[0], callbacks=[visualization], max_epochs=200) #Remove gpus=[0] to train using CPU only
trainer = Trainer(max_epochs=200)
# Add logger=comet_logger to log with comet.ml


mnist = MNISTDataModule(train_batch_size=100)  # Data
update_weights_rule = BioLearningRule(
    delta=0.40, lebesgue_p=2, ranking_param=2)  # Learning rule
net = Net(28**2, 10, num_hidden=2000)  # .double() #Model
print(net)
print(next(net.parameters()).dtype)


full_system = UnsupervisedPhase(
    model=net, update_weights_rule=update_weights_rule, n_epochs=trainer.max_epochs, lr=.02)  # Assemble!


trainer.fit(full_system, mnist)  # Run training!
# After every epoch, "Normalizations" are printed. These are mean and stdev of \sum_j |W_{ij}|^p.
# When the model converges, mean -> 1 (homeostatic constraint, with R=1)


torch.save(next(net.parameters()), "weights")


# Save model after training
now = datetime.now()
date_time_string = now.strftime("%d_%m_%Y-%H_%M")

checkpoint_name = f"../../data/project/checkpoints/Unsupervised-p{update_weights_rule.lebesgue_p}-k{update_weights_rule.ranking_param}-delta{update_weights_rule.delta}_{date_time_string}.ckpt"
# Move to models/biolearning_mnist_fully_connected
trainer.save_checkpoint(checkpoint_name)

print(f"Checkpoint saved to {checkpoint_name}")

torch.save(full_system.model.state_dict(
), '../../data/project/models/biolearning_mnist_fully_connected/unsupervised_weights_paper.pth')  # Save only state_dict
