"""Classify recorded stream failures against subsequently recovered artifacts."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.runtime;fixed=root/'orientation-correct-v2'
    patterns=[(root/'resume-logs',False),(fixed/'logs',True)]
    methods={'mv-zero','official-rgb1','trellis2','recon','mvadapter-recon','corradapter-recon','shapeup','spacecontrol'}
    issues={}
    for logdir,is_fixed in patterns:
        for log in logdir.glob('*.log'):
            match=re.match(r'^(.*?)-(post-\d+|render|score)(?:-retry|-restart)?\.log$',log.name)
            if not match or match.group(1) not in methods:continue
            method,kind=match.group(1),match.group(2)
            for line in log.read_text(errors='replace').splitlines():
                try:item=json.loads(line)
                except (ValueError,TypeError):continue
                if item.get('status') not in {'CASE_FAILED','RENDER_CASE_FAILED','FAILED'} or not item.get('sample_id'):continue
                sid=item['sample_id'];target=fixed if is_fixed else root
                if kind.startswith('post'):
                    resolved=any(json.loads(p.read_text()).get('status')=='PASS_FILE4_SHA256' for p in target.glob(f'{method}-postprocess-shard-*/cases/{sid}/archive.json'))
                elif kind=='render':resolved=(target/f'{method}-render-stream-v1/cases/{sid}/diagnostic-scores.json').is_file()
                else:resolved=(target/f'{method}-learned-metrics-v1/{sid}/learned.json').is_file() and (target/f'{method}-learned-metrics-v1/{sid}/inception.npz').is_file()
                key=(is_fixed,method,kind,sid)
                issues[key]={'corrected_run':is_fixed,'method':method,'stage':kind,'sample_id':sid,'resolved':resolved,'error':item.get('error'),'log':str(log)}
    values=list(issues.values());unresolved=[x for x in values if not x['resolved']]
    result={'status':'NO_UNRESOLVED_RECORDED_CASE_FAILURES' if not unresolved else 'UNRESOLVED_RECORDED_CASE_FAILURES','total_recorded':len(values),'resolved':len(values)-len(unresolved),'unresolved':unresolved,'all_issues':values,'scope':'structured per-case logs only; process liveness and full coverage require separate checks'}
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(a.output)
    print(json.dumps({k:v for k,v in result.items() if k!='all_issues'},ensure_ascii=False))
    if unresolved:raise SystemExit(1)

if __name__=='__main__':main()
