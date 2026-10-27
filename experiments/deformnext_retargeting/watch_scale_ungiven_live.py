"""Publish a read-only live ledger for the no-retraining TEST508 campaign."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time

PLAN = Path('/root/dev/.cache/deformnext-retargeting/scale-ungiven-test508-20261028/cascade-two-arm-plans-v1/campaign.json')
LOCAL = Path('/root/dev/.cache/deformnext-retargeting/scale-ungiven-test508-20261028')
BENCHMARK = Path('/root/dev/.cache/deformnext-benchmark-runtime')
REMOTE_QUERY = '''import json
from pathlib import Path
root=Path('/home/youngwoo/retargeting-unified-test508')
export=root/'official-scale-ungiven-export-common-success-v3-250k'
external=root/'source-sim3-final-evaluation-v6'
nine=root/'unified-nine-method-final-evaluation-v4'
ours=root/'ours-four-final-evaluation-v1'
surface=root/'official-scale-ungiven-source-sim3-common-success-v2/complete.json'
bridge=root/'official-fast-export-bridge-v3-recovered.json'
score=json.loads(surface.read_text()) if surface.is_file() else {}
metric_worker=None
bridge_worker=None
for process in Path('/proc').iterdir():
 if not process.name.isdigit():continue
 try:
  argv=(process/'cmdline').read_bytes().decode(errors='replace').split('\\0')
  state=(process/'stat').read_text().split(') ',1)[1][0]
  if (str(root/'bridge_official_fast_export.py') in argv
      and state not in ('Z','X')):
   bridge_worker=int(process.name)
  if (not argv or not argv[0].endswith('/M0Sim3-metrics')
      or str(external/'evaluation') not in argv):continue
  io=dict(line.split(': ',1) for line in (process/'io').read_text().splitlines())
  if state not in ('Z','X'):
   metric_worker={'pid':int(process.name),'state':state,
                  'read_bytes':int(io.get('read_bytes','0'))}
 except (OSError,ValueError,IndexError):
  continue
print(json.dumps({'official_surface':score.get('samples'),
 'official_export':len(list((export/'cases').glob('*/prediction.json'))),
 'official_export_done':(export/'complete.json').is_file(),
 'official_bridge_done':bridge.is_file(),
 'official_bridge_worker':bridge_worker,
 'external_metric_rows':len(list((external/'evaluation').glob('*-results.json'))),
 'external_done':(external/'evaluation/summary.json').is_file(),
 'external_metric_worker':metric_worker,
 'nine_done':(nine/'evaluation/summary.json').is_file(),
 'nine_archive_done':Path('/file4/youngwoo/deformation/experiments/retargeting-v1/source-sim3-test508-20261028/official-rgb1-scale-ungiven-common-success-v2/final-evaluation/completion.json').is_file(),
 'ours_metric_rows':len(list((ours/'evaluation-ours-four').glob('*-results.json'))),
 'ours_done':(ours/'evaluation-ours-four/summary.json').is_file()}))
'''


def local_counts() -> dict:
    campaign=json.loads(PLAN.read_text())
    counts={'v1':0,'selected':0,'selected_pbr':0,'pairs':0}
    by_id={}
    for entry in campaign['plans']:
        shape=json.loads(Path(entry['shape_plan']).read_text())
        done=(Path(shape['output'])/'archive-receipt.json').is_file()
        sid=entry['sample_id']
        by_id.setdefault(sid,{})[entry['arm']]=done
        if entry['arm']=='V1':
            counts['v1']+=done
        else:
            pbr=json.loads(Path(entry['top1_pbr_plan']).read_text())
            pbr_done=(Path(pbr['output'])/'archive-receipt.json').is_file()
            counts['selected']+=done
            counts['selected_pbr']+=pbr_done
            by_id[sid]['SELECTED'] &= pbr_done
    counts['pairs']=sum(all(v.get(k,False) for k in ('V1','SELECTED')) for v in by_id.values())
    counts['expected']=len(by_id)
    return counts


def remote_counts() -> dict:
    host='youngwoo@n16.unist.info'
    command='python3 -c '+shlex.quote(REMOTE_QUERY)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',
                           '-p','55555',host,command],capture_output=True,text=True,timeout=20,check=True)
    return json.loads(result.stdout)


def flag(path:Path)->bool:
    return path.is_file()


def local_worker_health()->dict[int,list[int]]:
    """Read live worker ranks; zombie children cannot advance the campaign."""
    result={rank:[] for rank in range(4)}
    module='experiments.deformnext_retargeting.run_minimal_test_cascade_worker'
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            argv=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if module not in argv or '--rank' not in argv:continue
            rank=int(argv[argv.index('--rank')+1])
            if rank not in result:continue
            if (argv[argv.index('--campaign')+1]!=str(PLAN)
                    or argv[argv.index('--world-size')+1]!='4'
                    or argv[argv.index('--physical-gpu')+1]!=str(rank)):
                continue
            state=(entry/'stat').read_text().split(') ',1)[1].split(' ',1)[0]
            if state not in {'Z','X'}:result[rank].append(int(entry.name))
        except (OSError,ValueError,IndexError):
            continue
    return result


def local_tail_health()->dict[int,list[tuple[int,int]]]:
    """Identify live disjoint tail ranks on physical GPUs 6 and 7."""
    result={gpu:[] for gpu in (6,7)}
    module='experiments.deformnext_retargeting.run_minimal_test_cascade_worker'
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            argv=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if module not in argv or '--physical-gpu' not in argv:continue
            gpu=int(argv[argv.index('--physical-gpu')+1])
            if gpu not in result:continue
            if (argv[argv.index('--campaign')+1]!=str(PLAN)
                    or argv[argv.index('--world-size')+1]!='508'):
                continue
            rank=int(argv[argv.index('--rank')+1])
            state=(entry/'stat').read_text().split(') ',1)[1].split(' ',1)[0]
            if state not in {'Z','X'}:result[gpu].append((int(entry.name),rank))
        except (OSError,ValueError,IndexError):
            continue
    return result


def local_tail_supervisor()->list[int]:
    """Track the live manager, not only its short-lived case children."""
    result=[]
    module='experiments.deformnext_retargeting.use_idle_gpus_for_test_tail'
    root=str(LOCAL/'disjoint-tail-gpu6-7-v3')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            argv=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if module not in argv or '--output-root' not in argv:continue
            if argv[argv.index('--output-root')+1]!=root:continue
            state=(entry/'stat').read_text().split(') ',1)[1].split(' ',1)[0]
            if state not in {'Z','X'}:result.append(int(entry.name))
        except (OSError,ValueError,IndexError):
            continue
    return result


def render(local:dict,remote:dict|None,error:str|None,
           health:dict[int,list[int]],tail:dict[int,list[tuple[int,int]]],
           tail_supervisor:list[int])->str:
    now=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    n=local['expected']
    lines=['# GT scale 미제공 TEST508 실시간 진행표', '',
           f'갱신: {now}. 이 표는 진행률이며 최종 성능표가 아니다.', '',
           '| 단계 | 검증된 진행률 |', '|---|---:|',
           f'| Ours V1 full cascade, file4 SHA | {local["v1"]}/{n} |',
           f'| Ours Selected full cascade, file4 SHA | {local["selected"]}/{n} |',
           f'| Ours Selected Top-1 PBR, file4 SHA | {local["selected_pbr"]}/{n} |',
           f'| 같은 사례의 두 arm 모두 완료 | {local["pairs"]}/{n} |']
    out='\n'.join(lines)
    if remote is not None:
        out+=f'\n| Official Pixal3D RGB1 source-M0 surface | {remote["official_surface"] or 0}/501 |'
        out+=f'\n| Official Pixal3D RGB1 render mesh | {remote["official_export"]}/501 |'
        bridge=('complete' if remote['official_bridge_done'] else
                f'waiting on external metrics (PID {remote["official_bridge_worker"]})'
                if remote.get('official_bridge_worker') else 'pending, no live bridge')
        out+=f'\n| Official render handoff | {bridge} |'
        out+=f'\n| External seven-method learned results | {remote["external_metric_rows"]}/7 |'
        worker=remote.get('external_metric_worker')
        detail=(f'{worker["pid"]}/{worker["state"]}/'
                f'{worker["read_bytes"]/(1024**3):.1f} GiB read'
                if worker else 'no active evaluator found')
        out+=f'\n| External learned evaluator (PID/state/I/O) | {detail} |'
        out+=f'\n| Unified nine-method learned evaluation | {"complete" if remote["nine_done"] else "pending"} |'
        out+=f'\n| Ours four-condition learned results | {remote["ours_metric_rows"]}/4 |'
    else:
        out+='\n| Node16 status | temporarily unavailable; local work continues |'
    active=', '.join(f'{rank}:{"/".join(map(str,pids)) if pids else "missing"}'
                     for rank,pids in health.items())
    out+=f'\n| Local primary cascade workers (rank: PID) | {active} |'
    tail_detail=', '.join(f'{gpu}:{"/".join(f"{pid}@{rank}" for pid,rank in rows) if rows else "idle"}'
                          for gpu,rows in tail.items())
    out+=f'\n| Local GPU 6/7 tail workers (GPU: PID@index) | {tail_detail} |'
    out+=f'\n| Local GPU 6/7 tail supervisor | {"/".join(map(str,tail_supervisor)) if tail_supervisor else "not running"} |'
    out+='\n\n'
    if error: out+=f'Remote read error: `{error[:300]}`. This does not mean the job failed.\n\n'
    if (remote is not None and not remote['external_done']
            and remote.get('external_metric_worker') is None):
        out+='Evaluator attention: no active external learned-metric process was found.\n\n'
    if (remote is not None and remote['external_done']
            and not remote['official_bridge_done']
            and remote.get('official_bridge_worker') is None):
        out+='Bridge attention: external metrics completed but no official render handoff process was found.\n\n'
    missing=[str(rank) for rank,pids in health.items() if not pids]
    if missing and not flag(LOCAL/'cascade-two-arm-execution-v2-direct-file4/complete.json'):
        out+=('Worker attention: primary rank(s) '+', '.join(missing)
              +' are absent while the cascade is incomplete; inspect recovery receipts.\n\n')
    if (not tail_supervisor
            and not flag(LOCAL/'disjoint-tail-gpu6-7-v3/complete.json')
            and not flag(LOCAL/'cascade-two-arm-execution-v2-direct-file4/complete.json')
            and (remote is None or not remote['official_bridge_done'])):
        out+='Tail attention: the GPU 6/7 manager ended without a handoff or completion receipt.\n\n'
    stages={
      'Ours cascade':LOCAL/'cascade-two-arm-execution-v2-direct-file4/complete.json',
      'Ours surface':LOCAL/'ours-surface-common502-v2/complete.json',
      'Official Cycles render':BENCHMARK/'official-rgb1-render-campaign-v2/complete.json',
      'Ours Cycles render':BENCHMARK/'ours-two-render-campaign-v2/complete.json',
      'Ours 10-metric evaluation':BENCHMARK/'ours-four-final-evaluation-monitor-v4/complete.json',
      'Common 10-method table':BENCHMARK/'scale-ungiven-main-table-v4/complete.json',
      'Ours paired uncertainty':BENCHMARK/'scale-ungiven-ours-paired-v3/complete.json',
      'Official RGB1 paired uncertainty':BENCHMARK/'scale-ungiven-official-contrasts-v2/complete.json',
      'Fair common-success teaser':LOCAL/'fair-final-teaser-v2/complete.json',
    }
    out+='| Completion gate | Receipt |\n|---|---|\n'
    for name,path in stages.items():out+=f'| {name} | {"PASS" if flag(path) else "pending"} |\n'
    out+='\nThe main table uses the final common-success intersection, camera-given where supported, no target-derived object scale at generation, and post-generation source-M0-only Sim(3). The older M1-derived-scale table is an internal diagnostic.\n'
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--interval-seconds',type=int,default=120)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if args.interval_seconds<60:raise ValueError('interval must be at least 60 seconds')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    while True:
        local=local_counts()
        try: remote=remote_counts();error=None
        except (OSError,subprocess.SubprocessError,json.JSONDecodeError) as exc:
            remote=None;error=str(exc)
        health=local_worker_health()
        tail=local_tail_health()
        supervisor=local_tail_supervisor()
        document=render(local,remote,error,health,tail,supervisor)
        temp=args.output.with_suffix(args.output.suffix+'.tmp')
        temp.write_text(document);temp.replace(args.output)
        print(json.dumps({'time':datetime.now(timezone.utc).isoformat(),
                          'local':local,'remote':remote,'worker_health':health,
                          'tail_health':tail,'tail_supervisor':supervisor,
                          'error':error}),flush=True)
        if (args.once
                or (flag(BENCHMARK/'scale-ungiven-ours-paired-v3/complete.json')
                    and flag(BENCHMARK/'scale-ungiven-official-contrasts-v2/complete.json')
                    and flag(LOCAL/'fair-final-teaser-v2/complete.json'))):
            break
        time.sleep(args.interval_seconds)

if __name__=='__main__':main()
