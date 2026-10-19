"""Explicit optimizer overrides for the candidate-attention parameters."""
import math

from torch import nn


def split_task_attention_group(groups, conditioner, epsilon):
    """Preserve existing groups, splitting only the declared task submodule."""
    if epsilon is None:
        return groups
    if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
            or not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError('task attention optimizer epsilon must be finite and positive')
    task = getattr(conditioner, 'task_attention', None)
    if not isinstance(task, nn.Module):
        raise ValueError('task attention optimizer epsilon requires the task module')
    candidates = [i for i, group in enumerate(groups)
                  if group.get('model_name') == 'deform_conditioner']
    if len(candidates) != 1 or any(group.get('model_name') == 'deform_conditioner.task_attention'
                                    for group in groups):
        raise ValueError('exactly one unsplit deform_conditioner group required')
    if any(not isinstance(group['params'], (list, tuple)) for group in groups):
        raise ValueError('explicit optimizer parameter lists required')
    all_ids = [id(p) for group in groups for p in group['params']]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('optimizer parameters appear in multiple groups')
    task_ids = {id(p) for p in task.parameters() if p.requires_grad}
    index = candidates[0]
    original = groups[index]
    original_ids = {id(p) for p in original['params']}
    if not task_ids or not task_ids <= original_ids:
        raise ValueError('all trainable task parameters must belong to the conditioner group')
    ordinary = [p for p in original['params'] if id(p) not in task_ids]
    attention = [p for p in original['params'] if id(p) in task_ids]
    if not ordinary:
        raise ValueError('conditioner must retain its non-task parameters')
    return [*groups[:index], {**original, 'params': ordinary},
            {**original, 'params': attention, 'model_name': 'deform_conditioner.task_attention',
             'eps': float(epsilon)}, *groups[index + 1:]]
