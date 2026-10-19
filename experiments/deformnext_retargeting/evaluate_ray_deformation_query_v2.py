"""Use the pinned DQ evaluator with the native SS quick-metric dispatcher."""


def main():
    from .deformation_query_metrics import install_metric_dispatch
    from .evaluate_ray_deformation_query import main as evaluate
    install_metric_dispatch()
    evaluate()


if __name__ == '__main__':
    main()
