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
REMOTE_QUERY = '''import json,os
from pathlib import Path
root=Path('/home/youngwoo/retargeting-unified-test508')
export=root/'official-scale-ungiven-export-common-success-v3-250k'
external=root/'source-sim3-final-evaluation-v6'
nine=root/'unified-nine-method-final-evaluation-v4'
ours=root/'ours-four-final-evaluation-v1'
precache=root/'ours-surface-precache-v1'
surface=root/'official-scale-ungiven-source-sim3-common-success-v2/complete.json'
bridge=root/'official-fast-export-bridge-v3-recovered.json'
score=json.loads(surface.read_text()) if surface.is_file() else {}
precache_ids=[]
for file in sorted(precache.glob('ids-*.json')):
 precache_ids.extend(json.loads(file.read_text()))
precache_cases=sum(len(path.stem)==64 and all(c in '0123456789abcdef' for c in path.stem)
 for path in precache.glob('*.json'))
precache_workers=[]
for file in sorted(precache.glob('*-launch.json')):
 for row in json.loads(file.read_text()).get('workers',[]):
  stat=Path(f'/proc/{row["pid"]}/stat')
  if stat.is_file():
   state=stat.read_text().split(') ',1)[1][0]
   if state not in ('Z','X'):precache_workers.append(row['pid'])
nine_ready=(nine/'evaluation/summary.json').is_file()
metric_worker=None
bridge_worker=None
cached_nine_worker=None
ours_binding_worker=None
ours_target_worker=None
ours_metric_worker=None
subset_file=root/'official-scale-ungiven-source-sim3-common-success-v2/official-success-common.json'
subset=json.loads(subset_file.read_text()).get('sample_ids',[]) if subset_file.is_file() else []
old_nine=Path('/proc/195556/stat')
old_nine_state=(old_nine.read_text().split(') ',1)[1][0]
 if old_nine.is_file() else 'missing')
for process in Path('/proc').iterdir():
 if not process.name.isdigit():continue
 try:
  argv=(process/'cmdline').read_bytes().decode(errors='replace').split('\\0')
  state=(process/'stat').read_text().split(') ',1)[1][0]
  if (str(root/'score_minimal_test_ours_target8.py') in argv
      and state not in ('Z','X')):
   io=dict(line.split(': ',1) for line in (process/'io').read_text().splitlines())
   ours_target_worker={'pid':int(process.name),'state':state,
                       'rchar':int(io.get('rchar','0'))}
  if (argv and argv[0].endswith('/M0Sim3-metrics')
      and str(ours/'evaluation-ours-four') in argv and state not in ('Z','X')):
   io=dict(line.split(': ',1) for line in (process/'io').read_text().splitlines())
   ours_metric_worker={'pid':int(process.name),'state':state,
                       'rchar':int(io.get('rchar','0'))}
  if (str(root/'build_minimal_test_ours_eval_manifests.py') in argv
      and state not in ('Z','X')):
   io=dict(line.split(': ',1) for line in (process/'io').read_text().splitlines())
   current=None
   for fd in (process/'fd').iterdir():
    try:
     parts=Path(os.readlink(fd)).parts
     current=next((subset.index(part)+1 for part in parts if part in subset),None)
     if current is not None:break
    except OSError:continue
   ours_binding_worker={'pid':int(process.name),'state':state,
                        'rchar':int(io.get('rchar','0')),
                        'current_case':current,'total':len(subset)}
  if (str(root/'bridge_official_fast_export.py') in argv
      and state not in ('Z','X')):
   bridge_worker=int(process.name)
  if (str(root/'orchestrate_official_scale_ungiven_evaluation_cached.py') in argv
      and state not in ('Z','X')):
   cached_nine_worker=int(process.name)
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
 'ours_surface_precache_cases':precache_cases,
 'ours_surface_precache_planned':len(precache_ids),
 'ours_surface_precache_workers':precache_workers,
 'external_metric_rows':len(list((external/'evaluation').glob('*-results.json'))),
 'external_done':(external/'evaluation/summary.json').is_file(),
 'external_metric_worker':metric_worker,
 'nine_done':nine_ready,
 'nine_cached_worker':cached_nine_worker,
 'nine_original_watcher_state':old_nine_state,
 'nine_archive_done':(nine_ready and Path('/file4/youngwoo/deformation/experiments/retargeting-v1/source-sim3-test508-20261028/official-rgb1-scale-ungiven-common-success-v2/final-evaluation/completion.json').is_file()),
 'ours_metric_rows':len(list((ours/'evaluation-ours-four').glob('*-results.json'))),
 'ours_binding_worker':ours_binding_worker,
 'ours_target_worker':ours_target_worker,
 'ours_metric_worker':ours_metric_worker,
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
    for name, directory in (
        ('official_render_chunks', BENCHMARK/'official-rgb1-render-campaign-v2/render-plan'),
        ('ours_render_chunks', BENCHMARK/'ours-two-render-campaign-v2/render-plan'),
    ):
        chunks=sorted(directory.glob('chunk[0-9][0-9][0-9]'))
        completed=0
        for chunk in chunks:
            receipt=chunk/'completed.json'
            if not receipt.is_file():continue
            try: row=json.loads(receipt.read_text())
            except json.JSONDecodeError:continue  # writer may be finishing the receipt
            completed+=(row.get('status')=='PASS_RENDER_CHUNK_ARCHIVED_ON_FILE4'
                        and row.get('chunk')==chunk.name)
        counts[name]={'completed':completed,'expected':len(chunks)}
    return counts


def remote_counts() -> dict:
    host='youngwoo@n16.unist.info'
    command='python3 -c '+shlex.quote(REMOTE_QUERY)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',
                           '-p','55555',host,command],capture_output=True,text=True,timeout=40,check=True)
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
    roots={str(LOCAL/f'disjoint-tail-gpu6-7-v{version}') for version in (3,4)}
    after_module='experiments.deformnext_retargeting.use_freed_gpus_for_test_gap'
    after_roots={str(LOCAL/name) for name in
                 ('post-official-disjoint-gap-v1', 'post-official-disjoint-gap-v2-gpu7')}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            argv=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if '--output-root' not in argv:continue
            output=argv[argv.index('--output-root')+1]
            if not ((module in argv and output in roots)
                    or (after_module in argv and output in after_roots)):continue
            state=(entry/'stat').read_text().split(') ',1)[1].split(' ',1)[0]
            if state not in {'Z','X'}:result.append(int(entry.name))
        except (OSError,ValueError,IndexError):
            continue
    return result


def local_tail_extension()->list[int]:
    result=[]
    module='experiments.deformnext_retargeting.orchestrate_disjoint_tail_extension'
    root=str(LOCAL/'disjoint-tail-gpu6-7-v4')
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
           tail_supervisor:list[int],tail_extension:list[int])->str:
    now=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    n=local['expected']
    lines=['# GT scale 미제공 TEST508 실시간 진행표', '',
           f'갱신: {now}. 이 표는 진행률이며 최종 성능표가 아니다.', '',
           '| 단계 | 검증된 진행률 |', '|---|---:|',
           f'| Ours V1 full cascade, file4 SHA | {local["v1"]}/{n} |',
           f'| Ours Selected full cascade, file4 SHA | {local["selected"]}/{n} |',
           f'| Ours Selected Top-1 PBR, file4 SHA | {local["selected_pbr"]}/{n} |',
           f'| 같은 사례의 두 arm 모두 완료 | {local["pairs"]}/{n} |']
    for label,key in (('Official RGB1 Cycles render chunks','official_render_chunks'),
                      ('Ours raw/+M0 Cycles render chunks','ours_render_chunks')):
        progress=local[key]
        value=(f'{progress["completed"]}/{progress["expected"]}'
               if progress['expected'] else 'pending render plan')
        lines.append(f'| {label}, file4 receipt | {value} |')
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
        cached=(f'PID {remote["nine_cached_worker"]}, old watcher {remote["nine_original_watcher_state"]}'
                if remote.get('nine_cached_worker') else
                f'not running, old watcher {remote.get("nine_original_watcher_state")}')
        out+=f'\n| Unified cached evaluator | {cached} |'
        out+=f'\n| Ours four-condition learned results | {remote["ours_metric_rows"]}/4 |'
        binding=remote.get('ours_binding_worker')
        if binding:
            case=binding['current_case']
            detail=(f'case {case}/{binding["total"]} (current file)' if case is not None
                    else 'current case unavailable')
            out+=f'\n| Ours input binding | PID {binding["pid"]}, {detail}, read {binding["rchar"] / 1e9:.2f} GB |'
        for label,key in (('Ours target image scoring','ours_target_worker'),
                          ('Ours learned metric evaluator','ours_metric_worker')):
            worker=remote.get(key)
            if worker:
                out+=f'\n| {label} | PID {worker["pid"]}, state {worker["state"]}, read {worker["rchar"] / 1e9:.2f} GB |'
        out+=f'\n| Ours source-M0 surface precache (independent of final score) | {remote["ours_surface_precache_cases"]}/{remote["ours_surface_precache_planned"]} cases; workers {remote["ours_surface_precache_workers"]} |'
    else:
        out+='\n| Node16 status | temporarily unavailable; local work continues |'
    active=', '.join(f'{rank}:{"/".join(map(str,pids)) if pids else "missing"}'
                     for rank,pids in health.items())
    out+=f'\n| Local primary cascade workers (rank: PID) | {active} |'
    tail_detail=', '.join(f'{gpu}:{"/".join(f"{pid}@{rank}" for pid,rank in rows) if rows else "idle"}'
                          for gpu,rows in tail.items())
    out+=f'\n| Local GPU 6/7 tail workers (GPU: PID@index) | {tail_detail} |'
    out+=f'\n| Local GPU 6/7 tail supervisor | {"/".join(map(str,tail_supervisor)) if tail_supervisor else "not running"} |'
    out+=f'\n| Local GPU 6/7 extension waiter | {"/".join(map(str,tail_extension)) if tail_extension else "not running"} |'
    out+='\n\n'
    if error: out+=f'Remote read error: `{error[:300]}`. This does not mean the job failed.\n\n'
    if (remote is not None and not remote['external_done']
            and remote.get('external_metric_worker') is None):
        out+='Evaluator attention: no active external learned-metric process was found.\n\n'
    if (remote is not None and remote['external_done']
            and not remote['official_bridge_done']
            and remote.get('official_bridge_worker') is None):
        out+='Bridge attention: external metrics completed but no official render handoff process was found.\n\n'
    if (remote is not None and not remote['nine_done']
            and remote.get('nine_original_watcher_state') == 'T'
            and remote.get('nine_cached_worker') is None):
        out+='Nine-method attention: the cached evaluator ended while the original watcher remains paused.\n\n'
    missing=[str(rank) for rank,pids in health.items() if not pids]
    if missing and not flag(LOCAL/'cascade-two-arm-execution-v2-direct-file4/complete.json'):
        out+=('Worker attention: primary rank(s) '+', '.join(missing)
              +' are absent while the cascade is incomplete; inspect recovery receipts.\n\n')
    active_tail=(LOCAL/'disjoint-tail-gpu6-7-v4' if (LOCAL/'disjoint-tail-gpu6-7-v4').exists()
                 else LOCAL/'disjoint-tail-gpu6-7-v3')
    if (not tail_supervisor
            and not flag(active_tail/'complete.json')
            and (active_tail.name.endswith('v4') or not flag(LOCAL/'disjoint-tail-gpu6-7-v3/complete.json'))
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
            remote=None
            error=(f'{type(exc).__name__}: remote status query timed out'
                   if isinstance(exc,subprocess.TimeoutExpired) else str(exc)[:300])
        health=local_worker_health()
        tail=local_tail_health()
        supervisor=local_tail_supervisor()
        extension=local_tail_extension()
        document=render(local,remote,error,health,tail,supervisor,extension)
        temp=args.output.with_suffix(args.output.suffix+'.tmp')
        temp.write_text(document);temp.replace(args.output)
        print(json.dumps({'time':datetime.now(timezone.utc).isoformat(),
                          'local':local,'remote':remote,'worker_health':health,
                          'tail_health':tail,'tail_supervisor':supervisor,
                          'tail_extension':extension,
                          'error':error}),flush=True)
        if (args.once
                or (flag(BENCHMARK/'scale-ungiven-ours-paired-v3/complete.json')
                    and flag(BENCHMARK/'scale-ungiven-official-contrasts-v2/complete.json')
                    and flag(LOCAL/'fair-final-teaser-v2/complete.json'))):
            break
        time.sleep(args.interval_seconds)

if __name__=='__main__':main()
