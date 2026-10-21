"""Explicit joint-source SS support around the sealed final508 campaign.

Existing model freeze, cache QA, split separation, sampler and metric guards
remain authoritative. This adapter routes only the declared new evaluator;
it neither selects a candidate nor creates a freeze or test cache receipt.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

from . import evaluate_bound_population as population
from . import frozen_campaign as campaign
from . import frozen_campaign_stage as stage
from . import prepare_frozen_ss_population as factory
from .evaluate_source_input_joint import validate_completed_joint_inputs

NAME='source_input_frozen_campaign'
JOINT='evaluate_source_input_joint'
SELF=Path(__file__).resolve()


def validate_joint_model(model,arguments):
    """Pin completed-run provenance in addition to the original model weights."""
    if model['evaluation_module']!=JOINT:return
    root=Path(population.argument(arguments,'--completed-training-root')).resolve()
    variant=population.argument(arguments,'--source-input-design')
    forbidden=['--source-teacher-init-spec','--pilot-variant','--conditioning-mode','--phase']
    if any(flag in arguments for flag in forbidden):
        raise ValueError('completed joint evaluator cannot use pilot initialization or an alternate phase/input')
    completion=model.get('joint_training_completion',{})
    if set(completion)!={'progress','startup','runtime'}:
        raise ValueError('freeze must pin actual joint progress/startup/source initialization report')
    names={'progress':'progress.json','startup':'startup.json','runtime':'source-input-joint-runtime.json'}
    for key,name in names.items():
        if population.pinned(completion[key]).resolve()!=root/name:
            raise ValueError('frozen joint completion file is outside its completed run')
    cfg,hashes=validate_completed_joint_inputs(root,population.pinned(model['arm_config']),
        population.pinned(model['checkpoint']),population.pinned(model['conditioner_checkpoint']),
        variant=variant,step=model['step'])
    if population.pinned(model['evaluator_source']).resolve()!=SELF.with_name(JOINT+'.py'):
        raise ValueError('explicit frozen joint SS evaluator source required')
    for spec in completion.values():
        if hashes[str(Path(spec['path']).resolve())]!=spec['SHA256']:
            raise ValueError('frozen joint completion bytes changed')
    # These files define the new constructor/loader and adapter. All original
    # numerical source bindings remain checked by the existing validator.
    package=SELF.parent;repo=package.parents[1]
    required=[SELF,package/(JOINT+'.py'),package/'source_teacher_initialization.py',
        package/'source_input_joint_runtime.py',
        repo/'pixal3d/modules/source_contrast_member_feature.py',
        repo/'pixal3d/modules/source_referenced_member_feature.py',
        repo/'pixal3d/modules/member_feature_source_ray.py']
    for path in required:
        if model['source_SHA256'].get(str(path))!=population.sha(path):
            raise ValueError('frozen new source-input runtime binding required: '+str(path))


def adapter_execution(execution):
    if population.pinned(execution['source_input_campaign_adapter']).resolve()!=SELF:
        raise ValueError('explicit pinned source-input campaign adapter required')


def prepare(plan,output):
    if plan.get('scope')!='final-test508':
        raise ValueError('source-input population factory requires the frozen final508 scope')
    freeze=json.loads(population.pinned(plan['model_freeze']).read_text())
    for composition in freeze['models'].values():
        model=composition['ss64']
        validate_joint_model(model,model.get('native_arguments',[]))
    with population_binding():return _PREPARE(plan,output)


def evaluate(plan_path):
    plan=json.loads(Path(plan_path).read_text())
    with population_binding():
        _,_,_,_,model=population.load_binding(plan)
        if model['evaluation_module']!=JOINT:
            raise ValueError('new SS population adapter requires the explicitly frozen joint evaluator')
        validate_joint_model(model,plan['native_arguments'])
        argv=sys.argv
        try:
            sys.argv=[population.__file__,'--plan',str(plan_path)];population.main()
        finally:sys.argv=argv


@contextmanager
def population_binding():
    original=population.MODULES
    if JOINT in original:raise ValueError('uncombined sealed native evaluator allow-list required')
    population.MODULES=original|{JOINT}
    try:yield
    finally:population.MODULES=original


def stage_command(plan,execution):
    if plan['kind']=='SS':
        native=json.loads(population.pinned(plan['native_plan']).read_text())
        if native.get('evaluation_module')==JOINT:
            adapter_execution(execution)
            if population.pinned(plan['entry_source']).resolve()!=SELF:
                raise ValueError('joint SS stage must pin its explicit population adapter')
            return [execution['python'],'-u','-m',campaign.PACKAGE+NAME,'SS-evaluate',
                    '--plan',plan['native_plan']['path']]
    return _STAGE_COMMAND(plan,execution)


def gpu_job(master,execution,*,name,kind,native,output,result_file,result_status,folder,depends):
    if kind!='SS' or json.loads(population.pinned(native).read_text()).get('evaluation_module')!=JOINT:
        return _GPU_JOB(master,execution,name=name,kind=kind,native=native,output=output,
            result_file=result_file,result_status=result_status,folder=folder,depends=depends)
    adapter_execution(execution)
    initial=json.loads(population.pinned(master['SS_plan']).read_text())
    spec=campaign.dump(folder/(name+'-job.json'),dict(scope='final-test508',kind=kind,native_plan=native,
        output=output,result_file=result_file,result_status=result_status,
        entry_source=dict(path=str(SELF),SHA256=population.sha(SELF)),model_freeze=initial['model_freeze']))
    return dict(name=name,resource='exclusive',priority=master['GPU_priority'],depends_on=depends,
        requires=[str(folder/'file4-closure.json')],outputs=[str(Path(output)/'file4-closure.json')],
        command=module_command(execution,'frozen_campaign_stage','--plan',spec['path'],'--plan-sha256',spec['SHA256'],
            '--execution',master['execution']['path'],'--execution-sha256',master['execution']['SHA256']))


def module_command(execution,module,*arguments):
    if module=='frozen_campaign':module=NAME
    elif module=='frozen_campaign_stage':module=NAME;arguments=('stage',*arguments)
    return _MODULE_COMMAND(execution,module,*arguments)


def load_master(spec):
    result=_LOAD_MASTER(spec);master,execution=result[:2];adapter_execution(execution)
    if master['source_SHA256'].get(str(SELF))!=population.sha(SELF):
        raise ValueError('campaign master must pin the new source-input adapter')
    return result


_PREPARE=factory.prepare
_STAGE_COMMAND=stage.command_for
_GPU_JOB=campaign.gpu_job
_MODULE_COMMAND=campaign.module_command
_LOAD_MASTER=campaign.load_master


@contextmanager
def campaign_binding():
    previous=(factory.prepare,stage.command_for,campaign.gpu_job,campaign.module_command,campaign.load_master)
    if previous!=(_PREPARE,_STAGE_COMMAND,_GPU_JOB,_MODULE_COMMAND,_LOAD_MASTER):
        raise ValueError('uncombined original campaign runtime required')
    try:
        factory.prepare=prepare;stage.command_for=stage_command;campaign.gpu_job=gpu_job
        campaign.module_command=module_command;campaign.load_master=load_master
        yield
    finally:
        factory.prepare,stage.command_for,campaign.gpu_job,campaign.module_command,campaign.load_master=previous


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('mode',choices=['SS-plans','SS-evaluate','stage','start','advance'])
    local,remaining=parser.parse_known_args()
    if local.mode=='SS-plans':
        p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        args=p.parse_args(remaining);result=prepare(json.loads(args.plan.read_text()),args.output)
        print(json.dumps(dict(status=result['status'],models=result['models'],actual_GPU_execution=False)),flush=True)
    elif local.mode=='SS-evaluate':
        p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);args=p.parse_args(remaining);evaluate(args.plan)
    else:
        argv=sys.argv
        try:
            if local.mode=='stage':
                execution=json.loads(population.pinned(dict(
                    path=population.argument(remaining,'--execution'),
                    SHA256=population.argument(remaining,'--execution-sha256'))).read_text())
                adapter_execution(execution)
            with campaign_binding():
                if local.mode=='stage':sys.argv=[stage.__file__,*remaining];stage.main()
                else:sys.argv=[campaign.__file__,local.mode,*remaining];campaign.main()
        finally:sys.argv=argv


if __name__=='__main__':main()
