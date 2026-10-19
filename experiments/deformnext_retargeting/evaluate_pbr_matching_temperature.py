"""One declared saved-weight PBR attention-temperature intervention."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8388608),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--matching-temperature',type=float,required=True)
    parser.add_argument('--source-input-receipt',type=Path,required=True)
    local,remaining=parser.parse_known_args()
    if not math.isfinite(local.matching_temperature) or local.matching_temperature<=0:
        raise ValueError('finite positive declared temperature required')
    from pixal3d import trainers
    from . import evaluate_pbr_streaming as entry
    bound={str(Path(__file__).resolve()):sha(__file__),str(Path(entry.__file__).resolve()):sha(entry.__file__)}
    receipt=json.loads(local.source_input_receipt.read_text())
    if receipt['status']!='PASS' or not receipt['local_equals_actual_file4_SHA256']:
        raise ValueError('actual file4 input weight receipt required')
    closed={row['local_path']:row for row in receipt['files']}
    for arg in ['--checkpoint','--conditioner-checkpoint']:
        path=remaining[remaining.index(arg)+1];row=closed[path]
        if sha(path)!=row['sha256'] or Path(path).stat().st_size!=row['bytes']:
            raise ValueError('saved PBR evaluation weights differ from actual file4')
        bound[path]=row['sha256']
    original=trainers.ImageConditionedProjSparseFlowMatchingCFGTrainer
    captured=[]
    class TemperatureEvaluationTrainer(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            module=self.deform_conditioner
            if (module is None or module.arm!='b3' or module.implementation!='r9-native-v5'
                    or module.task_attention is not None or module.candidate_aggregation!='weighted'):
                raise ValueError('declared saved B3 weighted PBR checkpoint required')
            self.original_matching_temperature=module.score_temperature
            module.score_temperature=local.matching_temperature
            captured.append(self)
    trainers.ImageConditionedProjSparseFlowMatchingCFGTrainer=TemperatureEvaluationTrainer
    previous=sys.argv
    try:
        sys.argv=[str(entry.__file__),*remaining]
        entry.main()
    finally:
        sys.argv=previous
        trainers.ImageConditionedProjSparseFlowMatchingCFGTrainer=original
    if len(captured)!=1 or any(sha(path)!=digest for path,digest in bound.items()):
        raise ValueError('weight/source binding or constructor count changed')
    trainer=captured[0]
    if trainer.deform_conditioner.score_temperature!=local.matching_temperature:
        raise ValueError('actual attention temperature changed unexpectedly')
    output=Path(remaining[remaining.index('--output')+1]);path=output/'evaluation.json'
    result=json.loads(path.read_text())
    result['diagnostic_only']=True
    result['matching_temperature_intervention']=dict(original_temperature=trainer.original_matching_temperature,
        actual_temperature=trainer.deform_conditioner.score_temperature,source_SHA256=bound,
        weights_unchanged_from_actual_file4=True,source_input_receipt=str(local.source_input_receipt),
        optimizer_updates=0,candidate_bank_unchanged=True,matching_decoder_unchanged=True,
        scope='Saved-weight temperature change on the existing diagnostic32 cohort;not retrained or an independent test.')
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
