"""Source-bound external algorithm execution; never substitute an internal method.

The detector port of an external algorithm supplies a subprocess runner and a
qualification report. This module verifies that implementation and its inputs,
executes an argv without a shell, and validates complete federation results.
It does not reinterpret a classification result as detector AP.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from .state import canonical, digest
from .storage import atomic_write
from .study import EXTERNAL
from .interventions import is_hash, validate_interventions

QUALIFICATION_CHECKS = ('algorithm_preserved', 'detector_loss_and_decode', 'D_only_fitting',
                        'V_only_selection', 'T_only_reporting', 'topology_and_history_declared',
                        'costs_and_failures_recorded', 'conditions_applied')


def _file(binding, inputs):
    if not isinstance(binding, dict) or set(binding) != {'path', 'sha256'} or not is_hash(binding['sha256']):
        raise ValueError('An exact external file path/SHA-256 binding is required')
    path = Path(binding['path']).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != binding['sha256']:
        raise ValueError('External file content changed: '+str(path))
    inputs[str(path)] = actual
    return path


def _under(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Source files must be relative to their explicit root')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('External source path escapes its root')
    return path


def source_identity(source):
    return digest({k: source[k] for k in ('repository_url', 'commit', 'files')})


def validate_adapter(adapter, job):
    """No imports of third-party code, model loading, subprocess training or sockets."""
    if not isinstance(adapter, dict) or adapter.get('schema') != 'cira.external_adapter.v1':
        raise ValueError('Explicit external adapter schema required')
    if adapter.get('method') not in EXTERNAL or adapter['method'] != job.get('method'):
        raise ValueError('External method identity differs from the requested algorithm')
    formal = job.get('formal', False)
    if type(formal) is not bool:
        raise ValueError('External formal flag must be boolean')
    inputs = {}
    for path, expected in job.get('input_manifest', {}).items():
        _file({'path': path, 'sha256': expected}, inputs)
    source = adapter['source']
    root = Path(source['root']).resolve()
    if not root.is_dir() or not isinstance(source.get('files'), dict) or not source['files']:
        raise ValueError('Bind the actual external source tree and file manifest')
    commit = source.get('commit')
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Pin a full external source commit')
    if not isinstance(source.get('repository_url'), str) or not source['repository_url'].startswith('https://'):
        raise ValueError('Declare the external source repository URL')
    for relative, checksum in source['files'].items():
        path = _under(root, relative)
        _file({'path': str(path), 'sha256': checksum}, inputs)
    # Bind executable code discovered beneath the checkout, not just the entry
    # script while leaving imported sibling modules mutable.
    executable_suffixes = {'.py', '.sh', '.ps1', '.toml', '.yaml', '.yml'}
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        if any(p in ('.git', '.venv', '__pycache__', 'data', 'datasets', 'runs', 'outputs') for p in relative.parts):
            continue
        if path.is_file() and path.suffix in executable_suffixes and relative.as_posix() not in source['files']:
            raise ValueError('Unbound executable source file: '+relative.as_posix())
    if formal:
        p = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=15)
        if p.returncode or p.stdout.strip() != commit:
            raise ValueError('Formal external checkout does not match its pinned commit')
    runner = adapter['runner']
    argv = runner.get('argv')
    if (not isinstance(argv, list) or len(argv) < 2 or
            any(not isinstance(x, str) or not x or '\0' in x for x in argv)):
        raise ValueError('Use an explicit argv; shell commands are not accepted')
    if not Path(argv[0]).is_absolute() or not Path(argv[0]).is_file():
        raise ValueError('Bind an absolute executable path')
    executable=Path(argv[0]).resolve()
    inputs[str(executable)]=hashlib.sha256(executable.read_bytes()).hexdigest()
    if Path(argv[0]).stem.lower() in ('sh', 'bash', 'cmd', 'powershell', 'pwsh'):
        raise ValueError('Use a direct executable/script argv rather than a shell')
    if not any('{job}' in x for x in argv) or not any('{result}' in x for x in argv):
        raise ValueError('External runner argv must consume {job} and produce {result}')
    for arg in argv:
        remaining = arg.replace('{job}', '').replace('{result}', '').replace('{output}', '')
        if '{' in remaining or '}' in remaining:
            raise ValueError('Unknown external runner placeholder')
    if any('softcas.production' in x or 'softcas.cli' in x for x in argv):
        raise ValueError('An external method cannot alias the CIRA executor')
    runner_files = runner.get('files', [])
    if not isinstance(runner_files, list) or not runner_files:
        raise ValueError('Bind the detection adaptation runner files')
    pinned_runner = {_file(b, inputs) for b in runner_files}
    scripts = [Path(x).resolve() for x in argv[1:] if Path(x).is_absolute() and Path(x).suffix == '.py']
    if not scripts or any(p not in pinned_runner for p in scripts):
        raise ValueError('Every Python entry script must have an explicit runner hash')
    cwd = Path(runner.get('cwd', root)).resolve()
    if not cwd.is_dir():
        raise ValueError('External working directory is unavailable')
    seconds = runner.get('timeout_seconds')
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('External runner needs a positive bounded timeout')
    environment = runner.get('environment', {})
    if not isinstance(environment, dict) or any(not isinstance(k,str) or not isinstance(v,str) or '\0' in k+v for k,v in environment.items()):
        raise ValueError('External environment overrides must be explicit strings')
    qualification_path = _file(adapter['qualification'], inputs)
    qualification = json.loads(qualification_path.read_bytes())
    if (qualification.get('schema') != 'cira.external_qualification.v1' or
            qualification.get('method') != adapter['method'] or
            qualification.get('source_identity') != source_identity(source)):
        raise ValueError('External qualification belongs to a different implementation')
    if formal:
        if qualification.get('validated') is not True or qualification.get('task') != 'object_detection':
            raise ValueError('Formal detector adaptation has not been qualified')
        if qualification.get('metric') != 'AP50:95' or not all(qualification.get('checks',{}).get(k) is True for k in QUALIFICATION_CHECKS):
            raise ValueError('External detector qualification checks are incomplete')
        evidence = qualification.get('evidence', [])
        if not evidence:
            raise ValueError('Bind actual qualification evidence; a boolean declaration is insufficient')
        for binding in evidence:
            _file(binding, inputs)
        if job.get('dataset') not in qualification.get('datasets',[]) or job.get('detector') not in qualification.get('detectors',[]):
            raise ValueError('External qualification does not cover this dataset/detector')
        if qualification.get('score_units') != 'fraction':
            raise ValueError('External AP is required in fraction units')
    validate_interventions(job.get('interventions'))
    return {'method': adapter['method'], 'source_identity': source_identity(source),
            'qualification_sha256': inputs[str(qualification_path)], 'formal': formal,
            'input_manifest': inputs, 'adapter_sha256': digest(adapter)}


def _validate_result(raw, job, directory):
    if raw.get('schema') != 'cira.external_trajectory.v1':
        raise ValueError('External runner did not return its trajectory contract')
    for key in ('method','seed'):
        if raw.get(key) != job[key]:
            raise ValueError('External result identity differs: '+key)
    if raw.get('cell') != job['dataset']+'/'+job['detector']:
        raise ValueError('External result has the wrong dataset/detector')
    for key in ('study_trajectory_id','study_recipe'):
        if key in job and raw.get(key) != job[key]:
            raise ValueError('External result has the wrong study recipe')
    if type(raw.get('complete')) is not bool:
        raise ValueError('Explicit external completion status required')
    if raw.get('input_manifest') != job.get('input_manifest', {}):
        raise ValueError('External runner must echo its exact bound input manifest')
    if raw.get('interventions_sha256') != digest(validate_interventions(job.get('interventions'))):
        raise ValueError('External runner did not identify the runtime conditions')
    result = copy.deepcopy(raw)
    result['formal']=job.get('formal',False)
    result['purpose']='FORMAL' if result['formal'] else 'EXTERNAL_SOFTWARE_VERIFICATION'
    if raw['complete']:
        formal = job.get('formal',False)
        if raw.get('score_units') != 'fraction' or (formal and raw.get('metric') != 'AP50:95'):
            raise ValueError('External detector scores have the wrong metric/units')
        if not formal and raw.get('metric') != 'test_quality_NOT_AP':
            raise ValueError('Unqualified software fixtures must retain NOT_AP identity')
        matrix_path = _file(raw['matrix'], {})
        if not matrix_path.is_relative_to(directory):
            raise ValueError('External result matrix must belong to this output directory')
        from .reporting import macro_endpoints
        matrix = json.loads(matrix_path.read_bytes())
        clients = tuple(map(str,job['clients']))
        if any(r.get('seed') != job['seed'] or str(r.get('client')) not in clients for r in matrix):
            raise ValueError('External matrix seed/client mismatch')
        if formal:
            if any(not is_hash(r.get('deployed')) for r in matrix):
                raise ValueError('External formal matrix needs deployed checkpoint identities')
            for key in {r['deployed'] for r in matrix}:
                binding=raw.get('deployed_artifacts',{}).get(key)
                if not binding or binding.get('sha256')!=key:
                    raise ValueError('External deployed identity lacks its exact checkpoint artifact')
                artifact=_file(binding,{})
                if not artifact.is_relative_to(directory):
                    raise ValueError('External deployed artifacts must belong to this run output')
            for client in clients:
                if {r['stage'] for r in matrix if str(r['client']) == client} != {1,2,3,4}:
                    raise ValueError('A complete external federation must include all four stages')
        endpoints = macro_endpoints(matrix,clients)
        final = max(endpoints,key=lambda r:r['stage'])
        for key in ('seen','current','forget','bwt','minimum_client_seen'):
            if raw.get(key) is not None and (final[key] is None or not math.isclose(raw[key],final[key],rel_tol=1e-10,abs_tol=1e-12)):
                raise ValueError('External summary differs from its deployed test matrix: '+key)
            result[key] = final[key]
        for field in ('F_total','network_tx_bytes','selection_E','max_client_update_seconds'):
            value=raw.get(field)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('Missing or invalid external resource field: '+field)
        if formal and (raw.get('computation_accounting_complete') is not True or raw.get('network_accounting_complete') is not True):
            raise ValueError('Complete formal external result needs complete resource accounting')
    else:
        for key in ('seen','current','forget','bwt','minimum_client_seen'):
            if raw.get(key) is not None:
                raise ValueError('Incomplete external trajectories must not fabricate endpoint scores')
            result[key]=None
    return result


def _stop_process(process):
    import psutil
    try:
        parent=psutil.Process(process.pid)
        children=parent.children(recursive=True)
        for child in children:
            try:child.terminate()
            except psutil.Error:pass
        process.terminate()
        _,alive=psutil.wait_procs(children,timeout=2)
        for child in alive:
            try:child.kill()
            except psutil.Error:pass
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:process.kill();process.wait()
    except psutil.NoSuchProcess:
        process.wait()


def execute_external(job, resume=False):
    from .study_execution import preflight_job
    if 'study_trajectory_id' in job:
        preflight_job(job)
    validation=validate_adapter(job['adapter'],job)
    root=Path(job['output']).resolve()
    identity=digest({'job':{k:v for k,v in job.items() if k!='output'},'validated_adapter':validation})
    statefile=root/'external_execution.json'
    if root.exists() and any(root.iterdir()):
        if not resume or not statefile.exists():
            raise FileExistsError('External output exists; use an identified completed resume or a new directory')
        saved=json.loads(statefile.read_bytes())
        if saved.get('identity') != identity:
            raise ValueError('External resume source/input/recipe identity changed')
        if saved.get('status') != 'complete':
            raise RuntimeError('Reconcile interrupted/failed external execution and its costs before a new attempt')
        resultpath=_file(saved['result'],{})
        return _validate_result(json.loads(resultpath.read_bytes()),job,root)
    root.mkdir(parents=True,exist_ok=True)
    jobfile,resultfile=root/'external_job.json',root/'external_result.json'
    contract={**job,'schema':'cira.external_job.v1','adapter_validation':validation,
              'result_path':str(resultfile),'resume':False,
              'required_output_schema':'cira.external_trajectory.v1'}
    atomic_write(jobfile,canonical(contract))
    atomic_write(statefile,canonical({'identity':identity,'status':'running','started':time.time()}))
    replace={'{job}':str(jobfile),'{result}':str(resultfile),'{output}':str(root)}
    argv=[]
    for arg in job['adapter']['runner']['argv']:
        for key,value in replace.items():arg=arg.replace(key,value)
        argv.append(arg)
    started=time.perf_counter();failure=None;code=None
    runner=job['adapter']['runner']
    with open(root/'runner_stdout.log','wb') as stdout,open(root/'runner_stderr.log','wb') as stderr:
        try:
            p=subprocess.Popen(argv,cwd=runner.get('cwd',job['adapter']['source']['root']),
                               env={**os.environ,**runner.get('environment',{})},stdin=subprocess.DEVNULL,
                               stdout=stdout,stderr=stderr,shell=False,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            try:code=p.wait(timeout=runner['timeout_seconds'])
            except subprocess.TimeoutExpired:
                _stop_process(p);failure='timeout'
            if code is not None and code!=0:failure='exit_status_'+str(code)
        except OSError as exc:failure=type(exc).__name__+': '+str(exc)
    elapsed=time.perf_counter()-started
    if failure is None:
        try:
            raw=json.loads(resultfile.read_bytes())
            result=_validate_result(raw,job,root)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            failure='invalid_result: '+str(exc)
    if failure is not None:
        failed={'identity':identity,'status':'failed','failure':failure,'returncode':code,
                'driver_elapsed_seconds':elapsed,'cost_complete':False}
        atomic_write(statefile,canonical(failed))
        result={**{k:job[k] for k in ('method','seed','study_trajectory_id','study_recipe') if k in job},
                'cell':job['dataset']+'/'+job['detector'],'complete':False,'failure':failure,
                'formal':job.get('formal',False),'purpose':'FORMAL' if job.get('formal',False) else 'EXTERNAL_SOFTWARE_VERIFICATION',
                'seen':None,'current':None,'forget':None,'bwt':None,'minimum_client_seen':None,
                'driver_elapsed_seconds':elapsed,'cost_complete':False}
        atomic_write(root/'trajectory.json',canonical(result))
        return result
    atomic_write(root/'trajectory.json',canonical(result))
    binding={'path':str(resultfile),'sha256':hashlib.sha256(resultfile.read_bytes()).hexdigest()}
    atomic_write(statefile,canonical({'identity':identity,'status':'complete' if result['complete'] else 'failed',
                                    'result':binding,'returncode':code,'driver_elapsed_seconds':elapsed}))
    return result
