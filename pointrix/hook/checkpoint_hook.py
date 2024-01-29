import os
from .hook import HOOK_REGISTRY, Hook


@HOOK_REGISTRY.register()
class CheckPointHook(Hook):

    def after_train_iter(self, trainner) -> None:
        """
        some operations after the training iteration ends.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """
        if trainner.global_step % 5000 == 0:
            trainner.point_cloud.save_ply(os.path.join(
                trainner.cfg.output_path, "{}.ply".format(trainner.global_step)))
