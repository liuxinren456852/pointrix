from .writer import TensorboardWriter, WandbWriter, LOGGER_REGISTRY, create_progress


def parse_writer(cfg, log_dir, **kwargs):
    writer_type = cfg.writer_type
    writer = LOGGER_REGISTRY.get(writer_type)
    return writer(log_dir, **kwargs)
