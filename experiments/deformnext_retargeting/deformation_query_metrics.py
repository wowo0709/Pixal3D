"""Select the existing SS metric implementation for the explicit DQ dataset.

The frozen native dispatcher uses dataset names, rather than SS inheritance.
Keep the real dataset/config identity and adapt only its metric-dispatch copy.
"""


def adapt_ss_metrics(evaluate, trainer, dataset, config):
    if config['dataset_name'] == 'DeformNeXtDeformationQuerySS64':
        if getattr(dataset, 'deformation_query_channels', None) != 2048:
            raise ValueError('physical RAW2048 dataset required for DQ SS metrics')
        return evaluate(trainer, dataset, dict(config, dataset_name='DeformNeXtSS64'))
    return evaluate(trainer, dataset, config)


def install_metric_dispatch():
    from . import quick_metrics
    original = quick_metrics.evaluate_quick
    if getattr(original, '_physical_DQ_dispatch', False):
        return

    def evaluate(trainer, dataset, config):
        return adapt_ss_metrics(original, trainer, dataset, config)

    evaluate._physical_DQ_dispatch = True
    quick_metrics.evaluate_quick = evaluate
