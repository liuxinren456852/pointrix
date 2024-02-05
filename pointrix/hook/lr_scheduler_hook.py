import os
from .hook import HOOK_REGISTRY, Hook


@HOOK_REGISTRY.register()
class LRSchedulerHook(Hook):
    """
    A hook to update the learning rate using the scheduler.
    """
    def before_train_iter(self, trainner) -> None:
        """
        some operations after the training iteration ends.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """
        if len(trainner.cfg.scheduler) > 0:
            for param_group in trainner.optimizer.param_groups:
                name = param_group['name']
                if name in trainner.schedulers.keys():
                    lr = trainner.schedulers[name](trainner.global_step)
                    param_group['lr'] = lr