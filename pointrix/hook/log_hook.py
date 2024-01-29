from .hook import HOOK_REGISTRY, Hook
from pointrix.logger.writer import Writer, create_progress, Logger


@HOOK_REGISTRY.register()
class LogHook(Hook):

    def __init__(self):
        self.ema_loss_for_log = 0.
        self.bar_info = {}

    def before_run(self, trainner) -> None:
        """
        some print operations before the training loop starts.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """
        try:
            Logger.print(" *************************************** ")
            Logger.print("The experiment name is {}".format(trainner.exp_dir))
            Logger.print(" *************************************** ")
        except AttributeError:
            Logger.print(
                "ERROR!!..Please provide the exp_name in config file..")

    def after_train_iter(self, trainner) -> None:
        """
        some operations after the training iteration ends.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """
        for key, value in trainner.loss_dict.items():
            if 'loss' in key:
                self.ema_loss_for_log = 0.4 * value.item() + 0.6 * self.ema_loss_for_log
                self.bar_info.update(
                    {key: f"{self.ema_loss_for_log:.{7}f}"})

            if trainner.logger and key != "optimizer_params":
                trainner.logger.add_scalar(key, value, trainner.global_step)

        if trainner.global_step % trainner.cfg.bar_upd_interval == 0:
            self.bar_info.update({
                "num_pt": f"{len(trainner.point_cloud)}",
            })
            trainner.progress_bar.set_postfix(self.bar_info)
            trainner.progress_bar.update(trainner.cfg.bar_upd_interval)
