import os
from .hook import HOOK_REGISTRY, Hook
from pointrix.logger.writer import Writer, create_progress, Logger


@HOOK_REGISTRY.register()
class LogHook(Hook):

    def __init__(self):
        self.ema_loss_for_log = 0.
        self.bar_info = {}

        self.l1_test = 0.
        self.psnr_test = 0.
        self.ssims_test = 0.
        self.lpips_test = 0.

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
            if name == "position":
                pos_lr = param_group['lr']
                break

        log_dict = {"loss": trainner.loss_dict['loss'],
                    "l1_loss": trainner.loss_dict['L1_loss'],
                    "ssim_loss": trainner.loss_dict['ssim_loss'],
                    "num_pt": len(trainner.point_cloud),
                    "pos_lr": pos_lr}

        for key, value in log_dict.items():
            if 'loss' in key:
                self.ema_loss_for_log = 0.4 * value.item() + 0.6 * self.ema_loss_for_log
                self.bar_info.update(
                    {key: f"{self.ema_loss_for_log:.{7}f}"})

            if trainner.logger and key != "optimizer_params":
                trainner.logger.write_scalar(key, value, trainner.global_step)

        if trainner.global_step % trainner.cfg.bar_upd_interval == 0:
            self.bar_info.update({
                "num_pt": f"{len(trainner.point_cloud)}",
            })
            trainner.progress_bar.set_postfix(self.bar_info)
            trainner.progress_bar.update(trainner.cfg.bar_upd_interval)

    def after_val_iter(self, trainner) -> None:
        self.l1_test += trainner.metric_dict['L1_loss']
        self.psnr_test += trainner.metric_dict['psnr']
        self.ssims_test += trainner.metric_dict['ssims']
        self.lpips_test += trainner.metric_dict['lpips']

        image_name = os.path.basename(trainner.metric_dict['rgb_file_name'])
        iteration = trainner.global_step
        trainner.logger.write_image(
            "test" + f"_view_{image_name}/render", 
            trainner.metric_dict['images'], 
            step=iteration)
        trainner.logger.write_image(
            "test" + f"_view_{image_name}/ground_truth", 
            trainner.metric_dict['gt_images'], 
            step=iteration)

    def after_val(self, trainner) -> None:
        self.l1_test /= trainner.val_dataset_size
        self.psnr_test /= trainner.val_dataset_size
        self.ssims_test /= trainner.val_dataset_size
        self.lpips_test /= trainner.val_dataset_size

        print(f"\n[ITER {trainner.global_step}] Evaluating test: L1 {self.l1_test:.5f} PSNR {self.psnr_test:.5f} SSIMS {self.ssims_test:.5f} LPIPS {self.lpips_test:.5f}")
        iteration = trainner.global_step
        trainner.logger.write_scalar(
            "test" + '/loss_viewpoint - l1_loss',
            self.l1_test,
            iteration
        )
        trainner.logger.write_scalar(
            "test" + '/loss_viewpoint - psnr',
            self.psnr_test,
            iteration
        )
        trainner.logger.write_scalar(
            "test" + '/loss_viewpoint - ssims',
            self.ssims_test,
            iteration
        )
        trainner.logger.write_scalar(
            "test" + '/loss_viewpoint - lpips',
            self.lpips_test,
            iteration
        )
        self.l1_test = 0.
        self.psnr_test = 0.
        self.ssims_test = 0.
        self.lpips_test = 0.
