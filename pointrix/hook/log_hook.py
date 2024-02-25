import os
from .hook import HOOK_REGISTRY, Hook
from pointrix.logger.writer import Writer, create_progress, Logger


@HOOK_REGISTRY.register()
class LogHook(Hook):
    """
    A hook to log the training and validation losses.
    """

    def __init__(self):
        self.ema_loss_for_log = 0.
        self.bar_info = {}
        
        self.losses_test = {"L1_loss": 0., "psnr": 0., "ssims": 0., "lpips": 0.}

    def before_train(self, trainner) -> None:
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
        for param_group in trainner.optimizer.param_groups:
            name = param_group['name']
            if name == "point_cloud." + "position":
                pos_lr = param_group['lr']
                break

        log_dict = {
            "num_pt": len(trainner.model.point_cloud),
            "pos_lr": pos_lr
        }
        log_dict.update(trainner.loss_dict)

        for key, value in log_dict.items():
            if 'loss' in key:
                self.ema_loss_for_log = 0.4 * value.item() + 0.6 * self.ema_loss_for_log
                self.bar_info.update(
                    {key: f"{self.ema_loss_for_log:.{7}f}"})

            if trainner.logger and key != "optimizer_params":
                trainner.logger.write_scalar(key, value, trainner.global_step)

        if trainner.global_step % trainner.cfg.bar_upd_interval == 0:
            self.bar_info.update({
                "num_pt": f"{len(trainner.model.point_cloud)}",
            })
            trainner.progress_bar.set_postfix(self.bar_info)
            trainner.progress_bar.update(trainner.cfg.bar_upd_interval)

    def after_val_iter(self, trainner) -> None:
        for key, value in trainner.metric_dict.items():
            if key in self.losses_test:
                self.losses_test[key] += value

        image_name = os.path.basename(trainner.metric_dict['rgb_file_name'])
        iteration = trainner.global_step
        trainner.logger.write_image(
            "test" + f"_view_{image_name}/render",
            trainner.metric_dict['images'].squeeze(),
            step=iteration)
        trainner.logger.write_image(
            "test" + f"_view_{image_name}/ground_truth",
            trainner.metric_dict['gt_images'].squeeze(),
            step=iteration)

    def after_val(self, trainner) -> None:
        
        log_info = f"\n[ITER {trainner.global_step}] Evaluating test:"

        for key in self.losses_test:
            self.losses_test[key] /= trainner.val_dataset_size
            trainner.logger.write_scalar(
                "test" + '/loss_viewpoint - ' + key,
                self.losses_test[key],
                trainner.global_step
            ) 
            log_info += f" {key} {self.losses_test[key]:.5f}"
        
        print(log_info)
        for key in self.losses_test:
            self.losses_test[key] = 0.
