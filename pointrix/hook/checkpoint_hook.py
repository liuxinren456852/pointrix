import os
import torch
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

    def after_train(self, trainner) -> None:
        """
        some operations after the training loop ends.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """
        data_list = {
            "global_step": trainner.global_step,
            "optimizer": trainner.optimizer.state_dict(),
            # "active_sh_degree": trainner.active_sh_degree, TODO: add this
            "point_cloud": trainner.point_cloud.state_dict(),
        }

        path = os.path.join(
            trainner.exp_dir, 
            "chkpnt" + str(trainner.global_step) + ".pth"
        )
        torch.save(data_list, path)