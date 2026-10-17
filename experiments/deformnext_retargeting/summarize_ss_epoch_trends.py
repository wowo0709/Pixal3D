"""Paired same-epoch diagnostics; never substitute these for final128 results."""
import json
import os
from pathlib import Path

from .compare_ss_noise_panels import compare


def completed_lines(path):
    text=path.read_text()
    return [json.loads(line) for line in text.splitlines(keepends=True) if line.endswith('\n') and line.strip()]


def record_epoch_contrasts(root):
    specifications=[('retarget-native-v3','p1-native-grid','p2-matcher-grid','p2-vs-p1'),
        *[('module-study-native-v4',left,right,right+'-vs-'+left) for left,right in [
            ('b2-top1','b2-top1-r'),('b2-top1-r','b3-attention-r'),
            ('b3-attention-r','c1-retarget-r'),('c1-retarget-r','c2-query-r')]]]
    written=[]
    for folder,left,right,name in specifications:
        stage=root/folder;left_run=stage/'runs_ss64'/left;right_run=stage/'runs_ss64'/right
        files=[r/'validation.jsonl' for r in [left_run,right_run]]
        if not all(f.is_file() for f in files):continue
        left_rows,right_rows=[{r['step']:r for r in completed_lines(f)} for f in files]
        configs=[json.loads((r/'config.json').read_text()) for r in [left_run,right_run]]
        steps=[c['validation']['quick_metrics']['sampling_steps'] for c in configs]
        if steps[0]!=steps[1]:raise ValueError('same-epoch sampling protocols differ')
        for step in sorted(left_rows.keys()&right_rows.keys()):
            if step==0 or step%96:continue
            out=stage/'epoch-comparisons'/f'epoch-{step//96:02d}-{name}.json'
            if out.exists():continue
            records=[]
            for row in [left_rows[step],right_rows[step]]:
                if len(row['quick_per_asset'])!=32:raise ValueError('same-epoch diagnostic requires the fixed32 panel')
                records.append({'per_asset':row['quick_per_asset'],'sampling_steps':steps[0],'guidance':3.})
            result=compare([(f'epoch{step//96}-common32',*records)])
            result.update(status='interim',epoch=step//96,step=step,baseline=left,candidate=right,
                selection_use=False,fixed_final128_three_noise=False,
                interpretation='same-epoch32 diagnostic; final adoption requires fixed-final128 with3 sampling repeats and review of training-seed uncertainty')
            out.parent.mkdir(exist_ok=True);temporary=out.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(result,indent=2)+'\n');os.replace(temporary,out);written.append(out)
    return written
