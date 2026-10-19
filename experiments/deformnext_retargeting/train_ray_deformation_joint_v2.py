"""Use the frozen DQ trainer with SS metric dispatch in every DDP child."""
from .train_ray_deformation_joint import worker as numerical_worker


def worker(local_rank, config):
    from .deformation_query_metrics import install_metric_dispatch
    install_metric_dispatch()
    numerical_worker(local_rank, config)


def main():
    from . import train_ray_deformation_joint as entry
    entry.worker = worker
    entry.main()


if __name__ == '__main__':
    main()
