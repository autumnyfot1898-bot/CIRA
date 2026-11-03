"""Build source/asset bindings and DEV locks from actual supplied artifacts.

These commands read local metadata and create new binding outputs. They neither
download data nor train models, and never replace missing calibration with zero
or invent a DEV winner from the manuscript's numerical-study parameters.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
from .state import canonical, digest, Structure, FitSettings
from .storage import atomic_write
from .study import GRID, SERIAL_RULES, build_study
from .study_execution import binding_template, condition_key, condition_identity, file_binding, compile_study
from .external import source_identity

NATIVE_LOCAL_CANDIDATES = {
    'N-L01': {'mask':[1,0,0,0,0,0],'ratio':1/16,'lr':.0001,'residual':.0001},
    'N-L02': {'mask':[0,1,0,0,0,0],'ratio':1/8,'lr':.0003,'residual':.001},
    'N-L03': {'mask':[0,0,1,0,0,0],'ratio':1/16,'lr':.001,'residual':.0001},
    'N-L04': {'mask':[0,0,0,1,0,0],'ratio':1/8,'lr':.0001,'residual':.001},
    'N-L05': {'mask':[0,0,0,0,1,0],'ratio':1/8,'lr':.0003,'residual':.0001},
    'N-L06': {'mask':[0,0,0,0,0,1],'ratio':1/16,'lr':.001,'residual':.001},
    'N-L07': {'mask':[0,1,0,1,0,0],'ratio':1/16,'lr':.0003,'residual':.001},
    'N-L08': {'mask':[0,0,1,0,1,0],'ratio':1/8,'lr':.0003,'residual':.0001},
}


def _bind(value, base):
    if isinstance(value, dict):
        path=Path(value['path'])
    elif isinstance(value,str) and value:
        path=Path(value)
    else:
        raise ValueError('Supply an actual artifact path')
    path=(base/path).resolve() if not path.is_absolute() else path.resolve()
    result=file_binding(path)
    if isinstance(value,dict) and value.get('sha256') != result['sha256']:
        raise ValueError('Supplied artifact hash differs: '+str(path))
    return result


def _read(binding):
    return json.loads(Path(binding['path']).read_bytes())


def lock_development(spec):
    """Select Local/Serial per cell and a shared controller Q using DEV only.

    Every declared candidate needs all three completed native DEV seed records.
    Missing/failed candidates block selection rather than favoring available
    winners. A record is an exported complete federation endpoint, not a test
    image or a best individual client.
    """
    base=Path(spec.get('base_directory','.')).resolve()
    cells=spec['cells'];seeds=(9001,9002,9003)
    protocol=spec.get('protocol','native_paper_r1000')
    if protocol not in ('native_paper_r1000','software_fixture'):
        raise ValueError('Unknown native DEV protocol')
    if not isinstance(cells,dict) or not cells:
        raise ValueError('Declare each dataset/detector cell and the complete Local candidate grid')
    inputs=[_bind(p,base) for p in spec['records']]
    rows=[_read(p) for p in inputs]
    indexed={}
    for row in rows:
        if (row.get('role')!='DEV' or row.get('study_kind')!='native_detector' or row.get('complete') is not True or
                row.get('seed') not in seeds or row.get('cell') not in cells or
                row.get('metric')!='AP50:95' or row.get('score_units')!='fraction'):
            raise ValueError('Only complete native DEV federation records may select settings')
        value=row.get('seen')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Invalid native DEV macro SeenAP')
        if row.get('selection_metric')!='final_macro_seen' or row.get('clients')!=4 or row.get('stages')!=4:
            raise ValueError('DEV selection must use the final complete four-client/four-stage macro endpoint')
        key=(row['cell'],row['family'],row['candidate'],row['seed'])
        if key in indexed:
            raise ValueError('Duplicate DEV candidate/seed record')
        if not isinstance(row.get('budget_identity'),str) or not row['budget_identity']:
            raise ValueError('Bind the shared DEV budget identity')
        for field in ('F_total','network_tx_bytes'):
            value=row.get(field)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('DEV tie-breaking requires actual '+field)
        indexed[key]=row
    expected=set();cell_settings={};serial={};summaries=[]
    def summaries_for(cell,family,candidate):
        keys=[(cell,family,candidate,s) for s in seeds]
        expected.update(keys)
        if any(k not in indexed for k in keys):
            raise ValueError('Missing or failed DEV coverage: '+str((cell,family,candidate)))
        values=[indexed[k]['seen'] for k in keys]
        result=(sum(values)/len(values),sum(indexed[k]['F_total'] for k in keys)/len(keys),
                sum(indexed[k]['network_tx_bytes'] for k in keys)/len(keys))
        summaries.append({'cell':cell,'family':family,'candidate':candidate,'mean_seen':result[0],
                          'mean_F':result[1],'mean_B':result[2]})
        return result
    q_scores={q:[] for q in list(GRID)[:7]}
    for cell,configuration in cells.items():
        choices=configuration['local_candidates']
        if not isinstance(choices,dict) or not choices:
            raise ValueError('Declare Local candidate parameter identities')
        for fixed in choices.values():
            Structure(tuple(fixed['mask']),fixed['ratio']);FitSettings(fixed['lr'],fixed['residual'])
        if protocol=='native_paper_r1000' and choices!=NATIVE_LOCAL_CANDIDATES:
            raise ValueError('Native paper DEV selection requires the eight prespecified N-L01--N-L08 candidates')
        local_scores={name:summaries_for(cell,'local',name) for name in choices}
        local=sorted(choices,key=lambda name:(-local_scores[name][0],local_scores[name][1],local_scores[name][2],name))[0]
        serial_scores={name:summaries_for(cell,'serial',name) for name in SERIAL_RULES}
        rule=sorted(SERIAL_RULES,key=lambda name:(-serial_scores[name][0],serial_scores[name][1],serial_scores[name][2],name))[0]
        serial[cell]=rule;cell_settings[cell]={'fixed':copy.deepcopy(choices[local]),'local_candidate':local,'serial_rule':rule}
        for q in q_scores:q_scores[q].append(summaries_for(cell,'controller',q))
        identities={r['budget_identity'] for r in rows if r['cell']==cell}
        if len(identities)!=1:
            raise ValueError('DEV candidates have unequal budget identities within '+cell)
    if set(indexed)!=expected:
        raise ValueError('Undeclared DEV candidates or cells must not enter selection')
    winner=sorted(q_scores,key=lambda q:(-sum(v[0] for v in q_scores[q])/len(q_scores[q]),
                                         sum(v[1] for v in q_scores[q])/len(q_scores[q]),q))[0]
    for row in rows:
        if row['family']=='serial' and row.get('beta_lr')!=GRID[winner]['beta_lr']:
            raise ValueError('Serial DEV rules must use the selected controller receiving rate')
    evidence=digest({'inputs':inputs,'cells':cells,'selection_metric':'final_macro_seen','seeds':seeds,'evidence_scope':protocol})
    selection={'controller_q':winner,'locked':True,'sha256':evidence,'serial_rules':serial}
    return {'schema':'cira.development_lock.v1','role':'DEV','evidence_scope':protocol,'seeds':list(seeds),'selection':selection,
            'cells':cell_settings,'candidate_grids':copy.deepcopy(cells),'inputs':inputs,'candidate_summaries':summaries,
            'tie_rule':'larger mean SeenAP; Local/Serial then lower F, B and ID; controller then lower F and ID',
            'selection_sha256_meaning':'canonical DEV input manifest, not a self-referential lock-file hash'}


def native_study(spec):
    """Register using the actual DEV winner instead of editing an example Q0."""
    definition=copy.deepcopy(spec)
    asset=_bind(definition['development_lock'],Path(spec.get('base_directory','.')).resolve())
    lock=_read(asset)
    if lock.get('schema')!='cira.development_lock.v1' or lock.get('role')!='DEV' or lock.get('selection',{}).get('locked') is not True:
        raise ValueError('A completed native DEV lock is required for native study registration')
    for binding in lock.get('inputs',[]):_bind(binding,Path('.').resolve())
    definition['development_selection']=lock['selection']
    study=build_study(definition)
    study['native_development_lock']=asset
    return study


def bind_study(study,spec):
    """Turn an explicit asset inventory into hash-bound jobs and blocker report."""
    base=Path(spec.get('base_directory','.')).resolve()
    bound=binding_template(study,spec.get('selection'))
    bound.update({k:copy.deepcopy(spec[k]) for k in ('output_root','clients','formal','physical_nodes','epoch_start_unix',
                                                  'trajectory_start_epochs','trajectory_spacing_seconds',
                                                  'external_coordinator_id','stage_schedule','require_all_ready') if k in spec})
    errors=[]
    def bind(value,where):
        try:return _bind(value,base)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            errors.append({'field':where,'reason':str(exc)})
            return {'path':None,'sha256':None}
    bound['development_lock']=bind(spec.get('development_lock'),'development_lock')
    for cell,target in bound['cells'].items():
        source=spec.get('cells',{}).get(cell,{})
        for field in ('config','checkpoint','detector_config'):
            target[field]=bind(source.get(field),f'cells.{cell}.{field}')
        target['overrides']=copy.deepcopy(source.get('overrides',{}))
        target['extra_inputs']=[bind(p,f'cells.{cell}.extra_inputs') for p in source.get('extra_inputs',[])]
        if source.get('cost_profile'):
            profile=bind(source['cost_profile'],f'cells.{cell}.cost_profile')
            target['overrides']['cost_profile']=profile['path'];target['extra_inputs'].append(profile)
        if source.get('feature_profile'):
            feature=bind(source['feature_profile'],f'cells.{cell}.feature_profile')
            target['extra_inputs'].append(feature)
            if feature['path']:
                values=_read(feature)
                if values.get('role')!='DEV' or values.get('schema')!='cira.feature_scales.v1' or values.get('cell')!=cell:
                    errors.append({'field':f'cells.{cell}.feature_profile','reason':'Feature scales need the matching native DEV identity'})
                else:
                    from .controller import FeatureEncoder
                    features=values['features']
                    FeatureEncoder(features['fields'],features['scales'])
                    if len(features['fields'])!=25:
                        raise ValueError('The native paper protocol requires exactly 25 locked features')
                    target['overrides']['features']=features
        for rounds in target['budgets_by_round']:
            target['budgets_by_round'][rounds]=bind(source.get('budgets_by_round',{}).get(rounds),f'cells.{cell}.budgets_by_round.{rounds}')
    for key,target in bound['conditions'].items():
        source=spec.get('conditions',{}).get(key)
        target['manifest']=bind(source.get('manifest') if isinstance(source,dict) else source,f'conditions.{key}')
    for method,path in spec.get('external_adapters',{}).items():
        asset=bind(path,'external_adapters.'+method)
        if asset['path']:
            value=_read(asset)
            bound['external_adapters'][method]=value.get('adapter',value)
            bound.setdefault('extra_inputs',[]).append(asset)
    plan=compile_study(study,bound)
    return {'schema':'cira.binding_report.v1','bindings':bound,'compiled_plan':plan,'inventory_errors':errors,
            'ready':not errors and not plan['blocked'] and bool(plan['jobs']),'executed':False}


def write_binding_outputs(report,directory):
    root=Path(directory).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError('Binding output must be a new or empty directory')
    root.mkdir(parents=True,exist_ok=True)
    for name,key in [('bindings.json','bindings'),('compiled_plan.json','compiled_plan')]:
        atomic_write(root/name,canonical(report[key]))
    atomic_write(root/'binding_report.json',canonical(report))
    return {'directory':str(root),'bindings':str(root/'bindings.json'),'plan':str(root/'compiled_plan.json'),
            'report':str(root/'binding_report.json'),'ready':report['ready'],'executed':False}


def external_source_snapshot(spec):
    """Hash the real checkout without executing it or asserting qualification."""
    root=Path(spec['source_root']).resolve()
    revision=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,timeout=15)
    if revision.returncode:
        raise ValueError('Use the actual pinned external source checkout')
    files={}
    for p in root.rglob('*'):
        rel=p.relative_to(root)
        if any(x in ('.git','.venv','__pycache__','data','datasets','runs','outputs') for x in rel.parts):continue
        if p.is_file() and p.suffix in ('.py','.sh','.ps1','.toml','.yaml','.yml','.json'):
            files[rel.as_posix()]=file_binding(p)['sha256']
    source={'root':str(root),'repository_url':spec['repository_url'],'commit':revision.stdout.strip(),'files':files}
    return {'source':source,'source_identity':source_identity(source),'executed':False}


def bind_external(spec):
    """Freeze the real upstream checkout/port; qualification remains evidence-bound."""
    from .external import validate_adapter
    base=Path(spec.get('base_directory','.')).resolve()
    source=external_source_snapshot(spec)['source']
    runner=copy.deepcopy(spec['runner'])
    runner['files']=[_bind(p,base) for p in runner['files']]
    adapter={'schema':'cira.external_adapter.v1','method':spec['method'],'source':source,
             'runner':runner,'qualification':_bind(spec['qualification'],base)}
    check_job={**spec.get('check_job',{}),'method':spec['method'],'formal':spec.get('formal',True)}
    validation=validate_adapter(adapter,check_job)
    return {'adapter':adapter,'validation':validation,'source_identity':source_identity(source),'executed':False}
