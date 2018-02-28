import itertools

import torch

from torchnlp.configurable import configurable

# TODO: Add increasing batch_size scheduler
# TODO: Remove Optimizer as it is not NLP do a PytorchUtils library


class Optimizer(object):
    """ The Optimizer class encapsulates torch.optim package and provides functionalities
    for learning rate scheduling and gradient norm clipping.
    Args:
        optim (torch.optim.Optimizer): optimizer object, the parameters to be optimized
            should be given when instantiating the object, e.g. torch.optim.SGD(params)
        max_grad_norm (float, optional): value used for gradient norm clipping,
            set 0 to disable (default 0)
    """

    @configurable
    def __init__(self, optim, max_grad_norm=0.0):
        self.optimizer = optim
        self.scheduler = None
        self.max_grad_norm = max_grad_norm
        self.zero_grad = self.optimizer.zero_grad

    def set_scheduler(self, scheduler):
        """ Set the learning rate scheduler.
        Args:
            scheduler (torch.optim.lr_scheduler.*): object of learning rate scheduler,
               e.g. torch.optim.lr_scheduler.StepLR
        """
        self.scheduler = scheduler

    def step(self):
        """ Performs a single optimization step, including gradient norm clipping if necessary. """
        if self.max_grad_norm and self.max_grad_norm > 0:
            params = itertools.chain.from_iterable(
                [group['params'] for group in self.optimizer.param_groups])
            torch.nn.utils.clip_grad_norm(params, self.max_grad_norm)
        self.optimizer.step()

    def update(self, loss, epoch):
        """ Update the learning rate if the criteria of the scheduler are met.
        Args:
            loss (float): The current loss.  It could be training loss or developing loss
                depending on the caller.  By default the supervised trainer uses developing
                loss.
            epoch (int): The current epoch number.
        """
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(loss)
        else:
            self.scheduler.step()
