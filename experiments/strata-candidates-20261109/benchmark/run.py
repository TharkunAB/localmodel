"""Compare DeepSeek variants with explicit routing state and guarded OMP workloads."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Any
from benchmark_paths import CODE, HERE, ROOT
sys.path.insert(0,str(ROOT))
import localmodel
from bootstrap import configure
from provenance import verify
from variants import SEED, add_arguments, arguments, read_variant, live_evidence, log_evidence, check_markers
PYTHON=Path(sys.executable)
GUARD=CODE/'memory_guard.py'

def save(path:Path,value:Any)->None:
    path.write_text(json.dumps(value,indent=2)+'\n')

def available()->dict[str,float]:
    info=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return {'linux_gib':int(info['MemAvailable'].split()[0])/2**20,'loadavg_1m':os.getloadavg()[0]}

def stop_worker(output:Path,env:dict[str,str])->None:
    """Stop only the OMP scope recorded by this case."""
    worker=output/'worker.json'
    if not worker.exists():return
    pid=int(json.loads(worker.read_text())['pid']);proc=Path('/proc')/str(pid)
    marker=os.fsencode('OMP_BENCH_ROOT='+str(output))
    if proc.exists() and (proc.stat().st_uid!=os.getuid() or marker not in (proc/'environ').read_bytes().split(b'\0')):
        raise RuntimeError('Recorded worker PID belongs to another process.')
    try:
        subprocess.run(['systemctl','--user','stop',f'omp-local-{pid}.scope'],env=env,timeout=15,check=False,
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        subprocess.run(['systemctl','--user','kill','--signal=SIGKILL',f'omp-local-{pid}.scope'],env=env,
                       timeout=5,check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    finally:
        if proc.exists():
            try:os.killpg(pid,signal.SIGTERM)
            except ProcessLookupError:pass

def request(body:dict[str,Any])->Any:
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req=urllib.request.Request('http://127.0.0.1:8899/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with opener.open(req,timeout=240) as response:return json.load(response)


def verify_edit_schema(body:dict[str,Any])->None:
    """Reject tools that cannot satisfy the unchanged workspace guard."""
    edits=[tool.get('function',{}) for tool in body.get('tools',[]) if tool.get('function',{}).get('name')=='edit']
    if len(edits)!=1:raise RuntimeError('Expected one edit tool in the captured request.')
    schema=edits[0].get('parameters',{});properties=schema.get('properties',{});required=set(schema.get('required',[]))
    keys={'i','path','old_string','new_string','replace_all'}
    if not {'path','old_string','new_string'}<=required or not set(properties)<=keys or any(properties.get(k,{}).get('type')!='string' for k in ['path','old_string','new_string']):
        raise RuntimeError('Captured edit schema is not the guarded ordinary replacement mode.')
    for key,expected in (('temperature',1.0),('top_p',0.95),('seed',SEED)):
        if body.get(key) is not None and body[key]!=expected:
            raise RuntimeError(f'The OMP request overrides the fixed comparison {key}.')


def read_routing_profile(path:Path,info:dict[str,Any])->tuple[bytes,dict[str,Any]]:
    """Use the same verified full-target routing counts for every arm."""
    if path.is_symlink() or not path.is_file() or path.name!='profile-after.bin':
        raise RuntimeError('Use the frozen full-target profile-after.bin.')
    path=path.resolve()
    expected=int(info.get('layer_count',40))*int(info.get('expert_count',384))*4
    if expected!=40*384*4 or path.stat().st_size!=expected:
        raise RuntimeError('The full routing profile must contain 40x384 uint32 counters (61440 bytes).')
    before=path.stat();data=path.read_bytes();after=path.stat()
    if len(data)!=expected or (before.st_dev,before.st_ino,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_mtime_ns,after.st_ctime_ns):
        raise RuntimeError('Routing profile changed while it was read.')
    case=json.loads((path.parent/'case-summary.json').read_text())
    source=json.loads((path.parent/'effective-manifest.json').read_text())
    def pins(value:dict[str,Any])->list[tuple[str,int,str]]:
        return [(item['name'],item['size'],item['sha256']) for item in value['files']]
    if case.get('status')!='completed' or case.get('model')!='deepseek' or not case.get('server_stopped'):
        raise RuntimeError('Routing provenance is not a completed, stopped full-target case.')
    origin=json.loads((HERE/'routing-baseline/origin.json').read_text())
    if hashlib.sha256(data).hexdigest()!=origin['profile_sha256'] or hashlib.sha256((path.parent/'effective-manifest.json').read_bytes()).hexdigest()!=case['manifest_sha256']:
        raise RuntimeError('Routing counts or their historical manifest differ from the frozen baseline.')
    if source['model_revision']!=info['model_revision'] or pins(source)!=pins(info):
        raise RuntimeError('Routing profile was collected with different weight pins.')
    return data,{'kind':'frozen completed-case routing','source':str(path),'source_case':path.parent.name,
        'sha256':hashlib.sha256(data).hexdigest(),'size':len(data),'model_revision':source['model_revision'],
        'source_all_quality_checks_passed':case.get('all_quality_checks_passed'),
        'note':'Collected from this benchmark workload. This is a trained-routing replay, not a cold model start.'}


def dspark_evidence(path:Path,offset:int=0)->dict[str,Any]:
    """Read loaded draft-layer allocation and native draft counters, not CLI intent."""
    with path.open('rb') as stream:
        stream.seek(offset if path.stat().st_size>=offset else 0)
        text=stream.read().decode('utf-8',errors='replace')
    text=text.rsplit('engine started:',1)[-1]
    lines=text.splitlines()
    allocation=[line for line in lines if re.search(r'^ds4: CUDA0 layers \[0, 42\]',line)]
    disabled=[line for line in lines if 'dspark' in line.lower() and any(word in line.lower() for word in ('not loaded','not supported','disabled'))]
    counters=[(int(a),int(b)) for a,b in re.findall(r'drafts (\d+) of (\d+) kept',text)]
    return {'loaded':bool(allocation) and not disabled,'load_evidence':allocation,'disabled_evidence':disabled,
            'native_logged_requests':len(counters),'native_accepted_including_warmup':sum(a for a,_ in counters),
            'native_offered_including_warmup':sum(b for _,b in counters),
            'load_evidence_note':'Native CUDA layer range0..42 proves the40target and3draft layers loaded on this single GPU. src/dsv41/ds4_model.cu:2853-2857,6425.',
            'counter_scope':'Native logged requests from this engine start, including the public warmup. OMP-only counters are recorded separately.'}

def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-source-tag', required=True)
    parser.add_argument('--expected-native-version', required=True)
    parser.add_argument('--model',choices=('deepseek-full',),required=True)
    parser.add_argument('--manifest',type=Path,help='Optional expected weight pins; the installed profile supplies runtime settings.')
    parser.add_argument('--routing-profile',type=Path,default=HERE/'routing-baseline/profile-after.bin',help='Frozen full-target routing baseline; every arm must use identical counters.')
    parser.add_argument('--dry-run',action='store_true',help='Validate bounds and print the public launch configuration without loading weights.')
    parser.add_argument('--name',required=True)
    parser.add_argument('--target',type=int,default=16384)
    parser.add_argument('--targets',type=int,nargs='+')
    parser.add_argument('--context',type=int,choices=(196608,),default=196608)
    parser.add_argument('--mem',type=int,default=64)
    parser.add_argument('--expert-ram-gb',type=int,choices=(48,),default=48)
    parser.add_argument('--speculative',choices=('none',),default='none')
    parser.add_argument('--draft-tokens',type=int,choices=(1,2,3,5),default=5)
    parser.add_argument('--cpu-threads',type=int,choices=(12,),default=12)
    parser.add_argument('--prefill-chunk',type=int,choices=(4096,),default=4096)
    parser.add_argument('--vram-reserve-mib',type=int)
    parser.add_argument('--gpu-expert-fraction',type=float)
    parser.add_argument('--dspark-conf',type=float,default=0.65)
    parser.add_argument('--engram-cache-mib',type=int,default=256)
    parser.add_argument('--engram-threads',type=int,choices=(16,),default=16)
    add_arguments(parser)
    args=parser.parse_args()
    targets=args.targets or [args.target]
    if targets!=sorted(set(targets)) or not targets or len(targets)>4 or any(t not in (16384,32768,65536,131072,184320) for t in targets):parser.error('Use at most four increasing 16K,32K,64K,128K,180K targets.')
    if any(t+4096>args.context for t in targets):parser.error('Each target must leave at least4096response tokens inside context capacity.')
    serial_lock=(HERE/'benchmark.lock').open('a')
    try:fcntl.flock(serial_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another arm of this benchmark is running.') from None
    if args.mem!=64:
        parser.error('Use the fixed 64 GiB cap.')
    if args.vram_reserve_mib is not None or args.gpu_expert_fraction is not None:
        parser.error('Preserve the standard GPU allocation settings in this matched comparison.')
    if not 0<=args.dspark_conf<=1 or args.engram_cache_mib!=256:
        parser.error('Use confidence0..1 and a256MiB Engram cache.')
    if Path(args.name).name!=args.name or args.name in ('.','..'):parser.error('Use a case name without path components.')
    out=HERE/args.name
    model_name=localmodel.deepseek_strata.NAME
    selected_manifest=(args.manifest or localmodel.deepseek_strata.manifest_path(model_name)).resolve()
    runtime_record=verify(localmodel.deepseek_strata.manifest(model_name),args.expected_source_tag,args.expected_native_version)
    variant=read_variant(args,localmodel.deepseek_strata.manifest(model_name))
    runtime_record['candidate_variant']=variant
    configure(selected_manifest,args.model,out/'slots',variant)
    initial_profile:bytes|None=None
    routing:dict[str,Any]={'kind':'absent at launch'}
    if args.routing_profile:
        initial_profile,routing=read_routing_profile(args.routing_profile,localmodel.deepseek_strata.manifest(model_name))
    cap=args.mem;experts=args.expert_ram_gb
    command=[str(PYTHON),str(CODE/'bootstrap.py'),'--controller-marker',str(ROOT/'localmodel.py'),'--expected-source-tag',args.expected_source_tag,
             '--expected-native-version',args.expected_native_version,'--manifest',str(selected_manifest),'--model',args.model,
             '--slot-directory',str(out/'slots'),*arguments(args),'--','start',model_name,'--mem',str(cap),
             '--expert-ram-gb',str(experts),'--expert-memory','cache','--prefill-chunk',str(args.prefill_chunk),'--context',str(args.context),
             '--reasoning-effort','xhigh','--speculative',args.speculative,'--draft-tokens',str(args.draft_tokens)]
    if args.model == 'deepseek-full':
        command += ['--cpu-threads',str(args.cpu_threads),'--engram-cache-mib',str(args.engram_cache_mib),'--engram-threads',str(args.engram_threads)]
        if args.speculative=='dspark':command += ['--dspark-conf',str(args.dspark_conf)]
        if args.vram_reserve_mib is not None:command += ['--vram-reserve-mib',str(args.vram_reserve_mib)]
        if args.gpu_expert_fraction is not None:command += ['--gpu-expert-fraction',str(args.gpu_expert_fraction)]
    if args.dry_run:
        subprocess.run([*command,'--dry-run'],check=True)
        return
    if not localmodel.deepseek_strata.verified(model=model_name):
        raise RuntimeError('Full target weights need the normal fetch/verification step before this run.')
    if args.speculative=='dspark' and not localmodel.deepseek_strata.verified(draft=True,model=model_name):
        raise RuntimeError('The existing DSpark draft needs normal checksum verification before this run.')
    if out.exists():raise RuntimeError('Case directory exists; use a new name.')
    if (ROOT/'state/active.json').exists():raise RuntimeError('Another managed model is running.')
    localmodel.check_free(8899)
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
    if gpu.stdout.strip():raise RuntimeError('A compute process is already using the GPU; do not overlap benchmark arms.')
    out.mkdir();headroom=available();save(out/'preflight.json',headroom)
    gpu_fields=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
    headroom.update(gpu_used_mib=float(gpu_fields[0]),gpu_load_percent=float(gpu_fields[1]))
    save(out/'preflight.json',headroom)
    if headroom['linux_gib']<72 or headroom['loadavg_1m']>8 or headroom['gpu_used_mib']>2048 or headroom['gpu_load_percent']>5:
        raise RuntimeError('Benchmark admission requires 72 GiB available RAM, CPU load <=8, GPU memory <=2 GiB and GPU load <=5%.')
    save(out/'runtime-provenance.json',runtime_record)
    save(out/'variant-requested.json',variant)
    save(out/'routing-profile.json',routing)
    if initial_profile is not None:(out/'profile-initial.bin').write_bytes(initial_profile)
    strata=localmodel.deepseek_strata
    with strata.model_lock():
        files:list[dict[str,Any]] = []
        for item in strata.manifest(model_name)['files']:
            path = strata.directory(model_name) / item['name']
            with path.open('rb') as handle:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            files.append({'name': path.name, 'size': path.stat().st_size})
        if args.speculative=='dspark':
            path=strata.drafter_directory(model_name)/strata.manifest(model_name)['dspark']['name']
            with path.open('rb') as handle:os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            files.append({'name':'drafter/'+path.name,'size':path.stat().st_size})
        save(out / 'weight-page-cache-release.json', {'advice': 'POSIX_FADV_DONTNEED', 'files': files})
    save(out/'effective-manifest.json',strata.manifest(model_name))
    for filename in ('run.py','bootstrap.py','provenance.py','variants.py','benchmark_paths.py','memory_guard.py','sample_metrics.py'):(out/('used-'+filename)).write_bytes((CODE/filename).read_bytes())
    save(out/'command.json',command)
    env=dict(os.environ,XDG_RUNTIME_DIR='/run/user/1000',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus',LD_LIBRARY_PATH='/usr/lib/wsl/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
    env.pop('CAPPED_UNIT',None)
    env['OMP_BENCH_MAX_REQUESTS']='12'
    env['PI_EDIT_VARIANT']='replace'
    env['HEAVY_PEAKS']=str(HERE/'peaks')
    os.environ.pop('CAPPED_UNIT',None)
    server:subprocess.Popen[str]|None=None;guard:subprocess.Popen[str]|None=None;observer:subprocess.Popen[str]|None=None
    report:dict[str,Any]={'status':'starting','model':args.model,'memory_cap_gib':cap,'linux_memory_floor_gib':8,
        'windows_memory_monitored':False,'speculative':args.speculative,'draft_tokens':args.draft_tokens,
        'target_input_tokens':targets[-1],'targets':targets,'server_context_capacity':args.context,'edit_mode':'replace','omp_timeout_seconds':1200,'case_timeout_seconds':1800,'preflight':headroom}
    profile=strata.pack_directory(args.speculative,model_name)/'expert-profile.bin'
    lane_profile=profile.with_name('lane-times.txt')
    state_config=ROOT/'state/deepseek-server.json'
    original_state_config=state_config.read_bytes() if state_config.exists() else None
    if original_state_config is not None:(out/'state-config-before.json').write_bytes(original_state_config)
    protected_settings=[Path.home()/'.omp/agent/config.yml',Path.home()/'.omp/agent/config.yaml',ROOT/'state/omp-local/agent/config.yml',ROOT/'state/omp-local/agent/config.yaml',ROOT/'state/omp-remote/agent/config.yml',ROOT/'state/omp-remote/agent/config.yaml']
    settings_before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in protected_settings}
    save(out/'normal-settings-before.json',settings_before)
    save(out/'active-state-before.json',{'active_existed':False,'server_config_existed':original_state_config is not None,'server_config_sha256':hashlib.sha256(original_state_config).hexdigest() if original_state_config is not None else None})
    report['workspace_guard_sha256']=runtime_record['workload_sha256']['experiments/deepseek-strata/omp-filled/workspace_guard.ts']
    original_lane:bytes|None=None
    original_profile:bytes|None=None
    profile_reset=False
    engine_log=ROOT/'state/deepseek-engine.log'
    log_offset=engine_log.stat().st_size if engine_log.exists() else 0
    report.update(runtime_provenance=runtime_record,manifest=str(selected_manifest),manifest_sha256=hashlib.sha256(selected_manifest.read_bytes()).hexdigest(),routing_profile=routing,resident_experts=False,slot_directory=str(out/'slots'),cpu_threads=args.cpu_threads,prefill_chunk=args.prefill_chunk,dspark_conf=args.dspark_conf,engram_cache_mib=args.engram_cache_mib,engram_threads=args.engram_threads)
    report.update(expert_ram_gb=args.expert_ram_gb,vram_reserve_mib=args.vram_reserve_mib,gpu_expert_fraction=args.gpu_expert_fraction)
    report['candidate_variant']=variant
    report['decoder_provenance']={'target_model':model_name,'target_repository':strata.manifest(model_name)['model_repository'],
        'target_revision':strata.manifest(model_name)['model_revision'],'target_files':strata.manifest(model_name)['files'],
        'pack_directory':str(strata.pack_directory(args.speculative,model_name)),
        'draft_artifact':strata.manifest(model_name)['dspark'] if args.speculative=='dspark' else None,
        'requested_decoder':args.speculative,'requested_draft_tokens':args.draft_tokens if args.speculative=='dspark' else None}
    start=time.monotonic();scope:Path|None=None
    signal.alarm(1800)
    try:
        with strata.model_lock():
            strata.pack_files(args.speculative,model_name)
            original_profile=profile.read_bytes() if profile.exists() else None
            save(out/'profile-before.json',{'existed':original_profile is not None,'sha256':hashlib.sha256(original_profile).hexdigest() if original_profile is not None else None})
            if original_profile is not None:(out/'profile-before.bin').write_bytes(original_profile)
            original_lane=lane_profile.read_bytes() if lane_profile.exists() else None
            save(out/'lane-before.json',{'existed':original_lane is not None,'sha256':hashlib.sha256(original_lane).hexdigest() if original_lane is not None else None})
            if original_lane is not None:(out/'lane-times-before.txt').write_bytes(original_lane)
            profile_reset=True
            if lane_profile.exists():lane_profile.unlink()
            if initial_profile is not None:profile.write_bytes(initial_profile)
            elif profile.exists():profile.unlink()
        with (out/'observer.log').open('w') as log:
            observer=subprocess.Popen([str(PYTHON),str(CODE/'sample_metrics.py'),'--seconds','1800'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,text=True)
        with (out/'server.log').open('w') as log:
            server=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,text=True)
        while not (ROOT/'state/active.json').exists():
            if server.poll() is not None or time.monotonic()-start>120:raise RuntimeError('Launcher failed before active record.')
            time.sleep(1)
        active=json.loads((ROOT/'state/active.json').read_text())
        if active['controller_pid']!=server.pid:raise RuntimeError('Active server is not this benchmark.')
        with (out/'guard.log').open('w') as log:
            guard=subprocess.Popen([str(PYTHON),str(GUARD),str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,text=True)
        while True:
            if server.poll() is not None or guard.poll() is not None or (out/'aborted.json').exists() or time.monotonic()-start>600:raise RuntimeError('Model did not become ready; read saved logs.')
            try:
                status=localmodel.current(8899)
                break
            except (OSError,ValueError,RuntimeError):time.sleep(2)
        if not (out/'memory-current.json').exists():raise RuntimeError('Memory guard did not report.')
        group=next(x[3:] for x in (Path('/proc')/str(active['pid'])/'cgroup').read_text().splitlines() if x.startswith('0::'))
        scope=Path('/sys/fs/cgroup'+group)
        limits={k:(scope/k).read_text().strip() for k in ('memory.max','memory.swap.max')}
        assert limits=={'memory.max':str(cap*2**30),'memory.swap.max':'0'},limits
        report.update(startup_seconds=time.monotonic()-start,status='ready',live=status,limits=limits)
        config_name='deepseek-server.json'
        launch=json.loads((ROOT/'state'/config_name).read_text())
        save(out/'launch-config.json',launch)
        report['variant_live']=live_evidence(active,variant,launch,engine_log,log_offset)
        save(out/'variant-live.json',report['variant_live'])
        report['strata_slots']=status.get('strata_slots')
        report['expert_memory']=status.get('expert_memory')
        if report['expert_memory']!='cache' or status.get('expert_ram_gb')!=args.expert_ram_gb:
            raise RuntimeError('Live expert memory mode or cache budget differs from the selected arm.')
        if status['model']!=model_name or status['context']!=args.context or status.get('cpu_threads')!=args.cpu_threads or status.get('prefill_chunk')!=args.prefill_chunk or status.get('speculative')!=args.speculative or (args.speculative=='dspark' and status.get('draft_tokens')!=args.draft_tokens):
            raise RuntimeError('Live model settings differ from the comparison plan.')
        if status.get('vram_reserve_mib')!=(args.vram_reserve_mib if args.vram_reserve_mib is not None else 700) or status.get('gpu_expert_fraction')!=args.gpu_expert_fraction or status.get('engram_threads')!=args.engram_threads or status.get('engram_cache_mib')!=args.engram_cache_mib:
            raise RuntimeError('Live GPU allocation controls differ from the comparison plan.')
        if args.speculative=='dspark' and status.get('dspark_conf')!=args.dspark_conf:
            raise RuntimeError('Live DSpark confidence differs from the comparison plan.')
        report['dspark_startup']=dspark_evidence(engine_log,log_offset)
        if report['dspark_startup']['loaded']!=(args.speculative=='dspark'):
            raise RuntimeError('Native draft-layer loading differs from the requested decoder.')
        save(out/'case-summary.json',report)
        print(json.dumps({'ready':args.model,'startup_seconds':report['startup_seconds']}),flush=True)
        body={'model':status['model'],'messages':[{'role':'user','content':'Return only the Python function clamp(x, lo, hi) which clamps x to inclusive bounds.'}],'max_tokens':128,'temperature':0,'reasoning_effort':'none'}
        save(out/'short-request.json',body);t=time.monotonic();response=request(body);save(out/'short-response.json',response)
        report['short_seconds']=time.monotonic()-t
        report['variant_after_warmup']=live_evidence(active,variant,launch,engine_log,log_offset)
        save(out/'variant-live-after-warmup.json',report['variant_after_warmup'])
        report['variant_warmup_markers']=log_evidence(engine_log,log_offset)
        check_markers(variant,report['variant_warmup_markers'],require_route=True)
        save(out/'variant-warmup-markers.json',report['variant_warmup_markers'])
        if args.speculative=='dspark':
            timing=response.get('timings',{})
            report['dspark_warmup']={'offered':timing.get('draft_n'),'accepted':timing.get('draft_n_accepted')}
            if not isinstance(timing.get('draft_n'),int) or timing['draft_n']<=0 or not isinstance(timing.get('draft_n_accepted'),int):
                raise RuntimeError('DSpark loaded but the warmup did not provide positive offered-draft evidence.')
        print(json.dumps({'short_completed':args.model,'seconds':report['short_seconds'],'usage':response.get('usage')}),flush=True)
        spec=importlib.util.spec_from_file_location('filled',ROOT/'experiments/deepseek-strata/filled_omp.py');assert spec and spec.loader and spec.origin
        filled=importlib.util.module_from_spec(spec)
        filled_source=Path(spec.origin).read_text()
        for old,new in (("if model not in (MODEL, 'qwen3.8'):","if model not in (*localmodel.deepseek_strata.NAMES, 'qwen3.8'):"),
                        ("localmodel.deepseek_strata.directory() / 'strata-pack/tokenizer'","localmodel.deepseek_strata.directory(model=model) / 'strata-pack/tokenizer'"),
                        ("if model == MODEL:","if model in localmodel.deepseek_strata.NAMES:")):
            if filled_source.count(old)!=1:raise RuntimeError('Filled OMP adapter changed; review its model selection before running.')
            filled_source=filled_source.replace(old,new)
        exec(compile(filled_source,spec.origin,'exec'),filled.__dict__)
        (out/'used-filled-omp.py').write_text(filled_source)
        os.environ.update(env)
        benchmark=filled.Benchmark(out,targets,1200,model=status['model'],context=status['context'])
        benchmark.facts={str(target):{field:prefix+'-'+(hashlib.sha256(f'deepseek029:{target}:{field}'.encode()).hexdigest()[:10] if len(targets)>1 else '026-benchmark') for field,prefix in (('release_key','release'),('target_branch','branch'),('deadline_tag','deadline'))} for target in targets}
        benchmark.report.update(memory_limit=f'{cap}G',expert_cache_gib=status['expert_ram_gb'],
            server_context_capacity=status['context'],engine_version=strata.manifest(model_name)['runtime']['version'],
            speculative=args.speculative,draft_tokens=args.draft_tokens,linux_only_memory_guard=True)
        if max(targets)>131072:
            extra=json.loads((HERE/'extended-corpus.json').read_text())
            extra_path=Path(extra['path'])
            benchmark.corpus+='\nPublic source file: '+extra['upstream_path']+'\n'+extra_path.read_text()
            benchmark.report['sources'].append(extra)
        original_prepare=benchmark.prepare
        def prepare()->None:
            original_prepare()
            config=json.loads((out/'config/omp.json').read_text())
            if config.get('edit.modelVariants') or config.get('edit',{}).get('modelVariants'):
                raise RuntimeError('A per-model edit-mode override could defeat the benchmark setting.')
        benchmark.prepare=prepare
        original_run_omp=benchmark.run_omp
        def run_omp(prompt:str,phase:str,capture:bool=False)->dict[str,Any]:
            if not capture and not (out/'edit-schema-check.json').exists():raise RuntimeError('Edit schema was not verified before OMP inference.')
            result=original_run_omp(prompt,phase,capture)
            if capture:
                verify_edit_schema(result)
                save(out/'edit-schema-check.json',{'mode':'replace','verified':True,'phase':phase})
            return result
        benchmark.run_omp=run_omp
        benchmark.report['edit_mode']='replace'
        benchmark.run()
        if guard.poll() is not None or (out/'aborted.json').exists():raise RuntimeError('The Linux memory guard did not remain active.')
        from collect import collect_case
        decoded=[collect_case(out,target) for target in targets]
        report['omp_draft_counters']=[{'target':target,'offered':item['draft_offered_tokens'],
            'accepted':item['draft_accepted_tokens'],'missing_request_counters':sum(
                request.get('draft_offered_tokens') is None or request.get('draft_accepted_tokens') is None
                for phase in item['phases'] for request in phase['requests'] if request['started'])}
            for target,item in zip(targets,decoded)]
        report['dspark_final']=dspark_evidence(engine_log,log_offset)
        report['variant_final_markers']=log_evidence(engine_log,log_offset)
        check_markers(variant,report['variant_final_markers'],require_route=True)
        save(out/'variant-final-markers.json',report['variant_final_markers'])
        if args.speculative=='dspark' and (not report['dspark_final']['loaded'] or any(
                item['missing_request_counters'] or not item['offered'] for item in report['omp_draft_counters'])):
            raise RuntimeError('DSpark did not retain positive load and draft-counter evidence through every checkpoint.')
        report.update(status='completed',all_quality_checks_passed=benchmark.report['all_quality_checks_passed'])
    except BaseException as error:
        report.update(status='failed',error=str(error));raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        try:
            stop_worker(out,env)
            report['worker_cleanup']='completed'
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
            report['worker_cleanup_error']=str(error)
        try:
            if scope is None and server is not None and (ROOT/'state/active.json').is_file():
                active=json.loads((ROOT/'state/active.json').read_text())
                if active['controller_pid']==server.pid:
                    group=next(x[3:] for x in (Path('/proc')/str(active['pid'])/'cgroup').read_text().splitlines() if x.startswith('0::'))
                    candidate=Path('/sys/fs/cgroup'+group)
                    if (candidate/'memory.max').read_text().strip()==str(cap*2**30):scope=candidate
            if scope is not None:report['memory']={k:(scope/k).read_text().strip() for k in ('memory.peak','memory.events','memory.swap.current') if (scope/k).is_file()}
        except (OSError,ValueError,KeyError,StopIteration) as error:report['memory_cleanup_error']=str(error)
        if server is not None:
            if server.poll() is None:
                try:os.kill(server.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            try:server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                try:
                    if scope is not None:subprocess.run(['systemctl','--user','kill','--signal=SIGKILL',scope.name],env=env,timeout=5,check=False)
                except (OSError,subprocess.SubprocessError) as error:
                    report['server_scope_cleanup_error']=str(error)
                finally:
                    server.kill();server.wait(timeout=10)
            report['server_stopped']=server.poll() is not None
        if guard is not None:
            if guard.poll() is None:guard.terminate()
            try:guard.wait(timeout=5)
            except subprocess.TimeoutExpired:guard.kill();guard.wait(timeout=5)
            report['guard_stopped']=guard.poll() is not None
        if observer is not None:
            if observer.poll() is None:observer.terminate()
            try:observer.wait(timeout=5)
            except subprocess.TimeoutExpired:observer.kill();observer.wait(timeout=5)
            report['observer_stopped']=observer.poll() is not None
        active_path=ROOT/'state/active.json'
        if server is not None and report.get('server_stopped') and active_path.exists():
            remaining=json.loads(active_path.read_text())
            if remaining.get('controller_pid')==server.pid and not Path('/proc',str(remaining['pid'])).exists():
                active_path.unlink()
                report['owned_stale_active_record_removed']=True
        report['port_8899_closed']=False
        try:
            localmodel.check_free(8899)
            report['port_8899_closed']=True
        except RuntimeError:pass
        if profile_reset and (server is None or report.get('server_stopped')) and not (ROOT/'state/active.json').exists():
            with strata.model_lock():
                if profile.exists():(out/'profile-after.bin').write_bytes(profile.read_bytes())
                if original_profile is not None:profile.write_bytes(original_profile)
                elif profile.exists():profile.unlink()
                report['profile_restored']=(profile.read_bytes()==original_profile if original_profile is not None else not profile.exists())
                if lane_profile.exists():
                    lane_bytes=lane_profile.read_bytes()
                    (out/'lane-times-after.txt').write_bytes(lane_bytes)
                    report['lane_calibration']={'size':len(lane_bytes),'sha256':hashlib.sha256(lane_bytes).hexdigest()}
                else:report['lane_calibration']=None
                if original_lane is not None:lane_profile.write_bytes(original_lane)
                elif lane_profile.exists():lane_profile.unlink()
                report['lane_profile_restored']=(lane_profile.read_bytes()==original_lane if original_lane is not None else not lane_profile.exists())
        if (server is None or report.get('server_stopped')) and not (ROOT/'state/active.json').exists():
            if original_state_config is not None:state_config.write_bytes(original_state_config)
            elif state_config.exists():state_config.unlink()
            report['state_config_restored']=(state_config.read_bytes()==original_state_config if original_state_config is not None else not state_config.exists())
            report['active_state_restored']=True
        else:report['active_state_restored']=False
        settings_after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in protected_settings}
        save(out/'normal-settings-after.json',settings_after)
        report['normal_settings_unchanged']=settings_after==settings_before
        if not report['normal_settings_unchanged']:
            report.update(status='failed',error='Normal OMP settings changed during the run; inspect hashes. No automatic global overwrite was attempted.')
        report['total_seconds']=time.monotonic()-start
        engine_log=ROOT/'state/deepseek-engine.log'
        if engine_log.exists():
            import shutil
            with engine_log.open('rb') as source, (out/'engine.log').open('wb') as dest:
                source.seek(log_offset if engine_log.stat().st_size>=log_offset else 0)
                shutil.copyfileobj(source,dest)
        save(out/'case-summary.json',report)
        print(json.dumps({k:report.get(k) for k in ('status','model','all_quality_checks_passed','server_stopped','profile_restored','total_seconds','error')}),flush=True)

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda _s,_f: (_ for _ in ()).throw(RuntimeError('Interrupted.')))
    signal.signal(signal.SIGALRM,lambda _s,_f: (_ for _ in ()).throw(TimeoutError('Case exceeded 30 minutes.')))
    main()
