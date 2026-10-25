"""Place native DEV cases across explicit training and auxiliary GPU queues.

Only scheduling metadata changes. An already trained common PBR model is a
numerical dependency; unfinished independent PBR training only occupies the
main queue's devices. Idle auxiliary devices can therefore evaluate cases.
"""
from copy import deepcopy

from .selected_pipeline_model_contract import require, validate_pbr_binding


def place_jobs(case_jobs, metric_jobs, *, main_training_job, common_PBR,
               main_devices, auxiliary_devices, prelaunch_receipt, activation_receipt):
    """Return one owner per128 case, with cross-queue artifact dependencies."""
    validate_pbr_binding(common_PBR, 'top1-only', seed=101)
    require(isinstance(main_devices,list) and isinstance(auxiliary_devices,list)
            and main_devices and auxiliary_devices,
            'explicit nonempty physical device lists required')
    devices = main_devices + auxiliary_devices
    require(len(set(devices)) == len(devices)
            and all(isinstance(d,str) and d.isdigit() for d in devices),
            'explicit nonempty disjoint physical device sets required')
    require(isinstance(main_training_job,str) and bool(main_training_job)
            and prelaunch_receipt != activation_receipt,
            'explicit resource dependency and distinct prelaunch/activation receipts required')
    require(len(case_jobs) == 128 and len(metric_jobs) == 3,
            'all128 own native cases and coverage/official/summary phases required')
    all_jobs = case_jobs + metric_jobs
    require(len({j['name'] for j in all_jobs}) == 131
            and all(len(j.get('outputs',[])) == 1 for j in all_jobs)
            and len({j['outputs'][0] for j in all_jobs}) == 131,
            'each case and metric phase requires a unique owner and output')
    main, auxiliary = [], []
    for i, job in enumerate(case_jobs):
        require(job.get('resource') == 'exclusive'
                and job.get('depends_on') == [main_training_job]
                and job.get('requires') == [prelaunch_receipt]
                and not any(k in job for k in ('gpu','gpus')),
                'original unpinned one-GPU case and resource-only main dependency required')
        row = deepcopy(job)
        row['requires'].append(activation_receipt)
        # Main devices remain reserved for the existing mandatory DDP run.
        # Auxiliary science/proof/evaluation priorities are10/20, above cases30.
        row['priority'] = 30
        if i % 2:
            row['depends_on'] = []
            auxiliary.append(row)
        else:
            main.append(row)
    coverage, official, summary = deepcopy(metric_jobs)
    require([r['resource'] for r in (coverage,official,summary)] == ['cpu','exclusive','cpu']
            and all(r.get('requires') == [prelaunch_receipt] for r in (coverage,official,summary))
            and all(not any(k in r for k in ('gpu','gpus')) for r in (coverage,official,summary)),
            'original CPU/GPU/CPU metric phases without implicit device assignment required')
    require(coverage.get('depends_on') == [j['name'] for j in case_jobs]
            and official.get('depends_on') == [coverage['name']]
            and summary.get('depends_on') == [official['name']],
            'original complete128 coverage and official-before-summary ordering required')
    coverage.update(depends_on=[], requires=[prelaunch_receipt,activation_receipt,
        *[j['outputs'][0] for j in case_jobs]], priority=20)
    official.update(depends_on=[], requires=[prelaunch_receipt,activation_receipt,
        coverage['outputs'][0]], priority=25)
    summary.update(depends_on=[], requires=[prelaunch_receipt,activation_receipt,
        official['outputs'][0]], priority=20)
    main.extend([coverage,summary]); auxiliary.append(official)
    return dict(main=main,auxiliary=auxiliary,
        main_devices=list(main_devices),auxiliary_devices=list(auxiliary_devices),
        main_case_count=64,auxiliary_case_count=64,
        actual_model_or_metric_execution=False,
        common_PBR_training_seed=101,unfinished_seed202_not_a_numerical_dependency=True)
