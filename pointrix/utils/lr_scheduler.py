import numpy as np

def get_expon_lr_func(
    init, final, delay_steps=0, delay_mult=0.01, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (init == 0.0 and final == 0.0):
            # Disable this parameter
            return 0.0
        if delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = delay_mult + (1 - delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(init) * (1 - t) + np.log(final) * t)
        return delay_rate * log_lerp

    return helper