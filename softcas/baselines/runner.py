"""Multi-stage native verification of independently implemented baseline ports.

This is a single-host logical federation runner. It preserves D/V/A/T roles and
algorithm state, but does not claim four-device operation or formal qualification.
The existing source/qualification-bound external executor remains the formal gate.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from ..data import detection_record_batch,validate_registry
from ..detector_baselines import LABELS,new_detector
from ..metering import ExecutedFlops
from ..state import canonical,digest
from ..storage import atomic_write
from .common import Batches,DetectorTask
from .factory import IMPLEMENTED,create


def file_binding(binding,name):
    if not isinstance(binding,dict) or not binding.get('path') or not binding.get('sha256'):
        raise ValueError('Fill '+name+'.path and '+name+'.sha256 in local configuration')
    path=Path(binding['path']).resolve()
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']:
        raise ValueError(name+' is missing or differs from its SHA-256 binding')
    return path


class ImageBatches:
    def __init__(self,records,root,size,batch_size,device):
        self.records,self.root,self.size,self.batch_size,self.device=tuple(records),root,size,batch_size,device
        if not self.records or type(batch_size) is not int or batch_size<1:raise ValueError('Nonempty role records and positive batch size required')

    def __len__(self):return (len(self.records)+self.batch_size-1)//self.batch_size

    def __getitem__(self,index):
        if isinstance(index,slice):return tuple(self[i] for i in range(*index.indices(len(self))))
        if not 0<=index<len(self):raise IndexError(index)
        records=self.records[index*self.batch_size:(index+1)*self.batch_size]
        values=[detection_record_batch(record,self.root,self.size) for record in records]
        batch={'img':torch.cat([value['img'] for value in values]),
               'batch_idx':torch.cat([value['batch_idx']+i for i,value in enumerate(values)]).long(),
               'cls':torch.cat([value['cls'] for value in values]),
               'bboxes':torch.cat([value['bboxes'] for value in values])}
        if any('auxiliary_target' in record for record in records):
            if not all(record.get('auxiliary_target') in (0,1) for record in records):
                raise ValueError('Explicit binary auxiliary targets required for every image in the batch')
            batch['auxiliary_target']=torch.tensor([record['auxiliary_target'] for record in records],dtype=torch.float32)
        return {name:value.to(self.device) for name,value in batch.items()}


def split(records,client,stage,role,root,size,batch_size,device,domain=None):
    selected=[]
    for record in records:
        if record['role']!=role:continue
        if role=='T':
            if str(record['client']) not in (str(client),'shared','*') or record['domain']!=domain:continue
        elif str(record['client'])!=str(client) or record['stage']!=stage:continue
        selected.append(record)
    if not selected:raise ValueError('Missing '+role+' support for client '+str(client)+' stage '+str(stage))
    if len({record['domain'] for record in selected})!=1:raise ValueError('One domain per requested split required')
    return Batches(role,ImageBatches(selected,root,size,batch_size,device))


def source_identity():
    root=Path(__file__).parent.parent
    return {path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob('*.py'))}


def validate_spec(spec):
    if spec.get('schema')!='cira.detector_port_run.v1' or spec.get('scope')!='NATIVE_ADAPTATION_VERIFICATION' or spec.get('formal') is not False:
        raise ValueError('This runner requires explicit non-formal NATIVE_ADAPTATION_VERIFICATION scope')
    if spec.get('method') not in IMPLEMENTED:raise ValueError('Choose one of the eight independently implemented algorithms')
    for field in ('device','output'):
        if not isinstance(spec.get(field),str) or not spec[field].strip():raise ValueError('Fill '+field+' in local configuration')
    for field in ('checkpoint','registry'):file_binding(spec.get(field),field)
    if type(spec.get('communication_iterations_per_stage')) is not int or spec['communication_iterations_per_stage']<1:
        raise ValueError('Set a positive algorithm communication/task-slot iteration count; no fitting-round count is guessed')
    clients=spec.get('clients',[]);stages=spec.get('stages',[])
    if not clients or len(set(map(str,clients)))!=len(clients) or not stages or stages!=sorted(set(stages)):
        raise ValueError('Unique client IDs and strictly ordered stage IDs required')
    if spec['method'] in ('pFedDIL','PerFedSDE-NAS') and spec.get('availability_by_stage'):
        raise ValueError('This server-reference port does not silently reinterpret peer-contact interventions')
    if 'seed' in spec.get('options',{}) and spec['options']['seed']!=spec.get('seed'):
        raise ValueError('Algorithm seed and registered trajectory seed differ')
    return spec


def _run(spec,resume=False):
    validate_spec(spec)
    root=Path(spec['output']).resolve();identity=digest({'spec':spec,'source':source_identity(),'torch':torch.__version__})
    checkpoint=root/'policy_checkpoint.pt';result_path=root/'detector_port_result.json'
    if root.exists() and any(root.iterdir()) and not resume:raise FileExistsError('Use a new output or explicit identified resume')
    root.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(spec['seed'])
    document=json.loads(file_binding(spec['registry'],'registry').read_text(encoding='utf-8'))
    if tuple(document['labels'])!=LABELS:raise ValueError('Registry labels differ from the fixed detector vocabulary')
    validate_registry(document['records'],False)
    records=document['records'];registry_root=Path(spec['registry']['path']).resolve().parent
    for record in records:
        path=Path(record['image']);path=path if path.is_absolute() else registry_root/path
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=record['image_sha256']:
            raise ValueError('Registry image is missing or changed before native execution')
    n=len(spec['clients']);device=spec['device'];size=spec['image_size']
    models=[new_detector(spec['detector'],device,checkpoint=spec['checkpoint']['path'],seed=spec['seed']) for _ in range(n)]
    options=copy.deepcopy(spec['options'])
    if spec['method']=='DA-DPFL' and options.get('total_rounds') is None:
        options['total_rounds']=len(spec['stages'])*spec['communication_iterations_per_stage']
    if spec['method']=='PerFedSDE-NAS':options.setdefault('image_size',size)
    first_stage=spec['stages'][0]
    if spec['method'] in ('DPFL','DCFCL','pFedDIL','PerFedSDE-NAS') and 'sample_weights' not in options:
        options['sample_weights']=[sum(record['role']=='D' and str(record['client'])==str(client) and record['stage']==first_stage for record in records) for client in spec['clients']]
    policy=create(spec['method'],models,options,spec.get('assets'))
    task=DetectorTask();matrix=[];diagnostics=[];seen={str(client):[] for client in spec['clients']}
    start_stage=start_iteration=0;F_total=0;unknown={}
    if resume:
        if not checkpoint.is_file():raise ValueError('No identified checkpoint to resume')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['identity']!=identity:raise ValueError('Resume specification/source changed')
        policy.load_state_dict(saved['policy']);matrix=saved['matrix'];diagnostics=saved['diagnostics'];seen=saved['seen']
        start_stage,start_iteration=saved['stage_index'],saved['next_iteration'];F_total=saved['F_total'];unknown=saved['unknown']
        torch.random.set_rng_state(saved['torch_rng'])
        if saved['cuda_rng'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(saved['cuda_rng'])
    report={'schema':'cira.detector_port_result.v1','scope':spec['scope'],'formal':False,'method':spec['method'],
            'detector':spec['detector'],'dataset':spec['dataset'],'seed':spec['seed'],'identity':identity,
            'status':'running','topology':'single_host_logical_federation','metric':'AP50:95','score_units':'fraction',
            'formal_experiment_complete':False,'physical_device_measurements':None,'source_training_verified':False}

    def save(stage_index,next_iteration):
        temporary=checkpoint.with_suffix('.pending.pt')
        torch.save({'identity':identity,'policy':policy.state_dict(),'matrix':matrix,'diagnostics':diagnostics,'seen':seen,
                    'stage_index':stage_index,'next_iteration':next_iteration,'F_total':F_total,'unknown':unknown,
                    'torch_rng':torch.random.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},temporary)
        temporary.replace(checkpoint)

    begun=time.perf_counter()
    try:
        for stage_index,stage in enumerate(spec['stages']):
            if stage_index<start_stage:continue
            data=[split(records,client,stage,'D',registry_root,size,spec['fit_batch_size'],device) for client in spec['clients']]
            selection=[split(records,client,stage,'V',registry_root,size,spec['evaluation_batch_size'],device) for client in spec['clients']]
            audit=[split(records,client,stage,'A',registry_root,size,spec['evaluation_batch_size'],device) for client in spec['clients']]
            available=spec.get('availability_by_stage',{}).get(str(stage))
            continuing=stage_index==start_stage and start_iteration>0
            if not continuing:
                if spec['method']=='DPFL' and policy.graph is None:diagnostics.append({'stage':stage,'preparation':policy.prepare(data,selection,available)})
                if hasattr(policy,'begin_stage'):
                    details=policy.begin_stage(data,stage=stage) if spec['method']=='PerFedSDE-NAS' else policy.begin_stage(data)
                    diagnostics.append({'stage':stage,'begin':details})
            counter=ExecutedFlops(strict=spec.get('meter',{}).get('strict',False),overrides=spec.get('meter',{}).get('overrides',{}))
            with counter:
                for iteration in range(start_iteration if continuing else 0,spec['communication_iterations_per_stage']):
                    details=policy.round(data,selection,available)
                    diagnostics.append({'stage':stage,'iteration':iteration,'algorithm':details})
                    # Persist all algorithm state, including graph/population,
                    # sparse masks, portfolios and signature caches.
                    F_total+=counter.total;counter.total=0
                    for name,count in counter.unknown.items():unknown[name]=unknown.get(name,0)+count
                    counter.unknown.clear()
                    save(stage_index,iteration+1)
            for i,client in enumerate(spec['clients']):
                domain=next(record['domain'] for record in records if record['role']=='D' and str(record['client'])==str(client) and record['stage']==stage)
                if domain not in seen[str(client)]:seen[str(client)].append(domain)
                predict=(lambda images,i=i:policy.predict(i,images)) if hasattr(policy,'predict') else None
                # A is an independent diagnostic, never an optimizer or model
                # selector. The verification runner does not invent a release rule.
                audit_score=task.ap(policy.models[i],audit[i],role='A',predict=predict)
                diagnostics.append({'stage':stage,'client':str(client),'A_diagnostic_AP':audit_score})
                artifact=root/('client_'+str(client)+'_stage_'+str(stage)+'.pt')
                if spec['method']=='pFedDIL':local_state=policy.portfolios[i].state_dict()
                elif spec['method']=='dFLMoE':
                    local_state={'state':copy.deepcopy(policy.clients[i].state_dict()),'active_experts':policy.clients[i].active}
                elif spec['method']=='DeModCL':
                    local_state={'modules':[module.state_dict() for module in policy.pools[i]],
                                 'extractor':copy.deepcopy(policy.models[i].state_dict())}
                else:
                    local_state={'state':copy.deepcopy(policy.models[i].state_dict()),
                                 'architecture':getattr(policy.models[i],'architecture',None)}
                torch.save({'method':spec['method'],'client':str(client),'detector':spec['detector'],
                            'scope':spec['scope'],'local_deployment':local_state},artifact)
                deployed=hashlib.sha256(artifact.read_bytes()).hexdigest()
                for old_domain in seen[str(client)]:
                    test=split(records,client,stage,'T',registry_root,size,spec['evaluation_batch_size'],device,old_domain)
                    score=task.ap(policy.models[i],test,role='T',predict=predict)
                    matrix.append({'seed':spec['seed'],'client':str(client),'stage':stage,'current_domain':domain,
                                   'domain':old_domain,'deployed':deployed,'score':score})
            start_iteration=0
            save(stage_index+1,0)
        report['status']='complete_native_verification'
    except Exception as error:
        report['status']='failed';report['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        report.update(elapsed_wall_seconds=time.perf_counter()-begun,executed_tensor_FLOPs=F_total,
                      unknown_arithmetic_operators=unknown,computation_accounting_complete=False,
                      network_accounting_complete=False,network_tx_bytes=None,
                      source_files=source_identity(),completed_matrix_rows=len(matrix))
        atomic_write(root/'matrix.json',canonical(matrix))
        atomic_write(root/'diagnostics.json',canonical(diagnostics))
        atomic_write(result_path,canonical(report))
        with (root/'attempts.jsonl').open('ab') as stream:
            stream.write(canonical({'identity':identity,'status':report['status'],'error':report.get('error'),
                                    'elapsed_wall_seconds':report['elapsed_wall_seconds'],'resumed':resume})+b'\n')
    return report


def run(spec,resume=False):
    output=spec.get('output')
    existed=bool(output) and Path(output).exists()
    try:
        return _run(spec,resume)
    except Exception as error:
        if output and not existed and Path(output).is_dir() and not (Path(output)/'detector_port_result.json').exists():
            atomic_write(Path(output)/'initialization_failure.json',canonical({'scope':'NATIVE_ADAPTATION_VERIFICATION',
                'status':'failed_before_training','error':type(error).__name__+': '+str(error)}))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True);parser.add_argument('--resume',action='store_true');parser.add_argument('--check',action='store_true')
    args=parser.parse_args();spec=json.loads(Path(args.spec).read_text(encoding='utf-8'))
    if args.check:
        validate_spec(spec);print(json.dumps({'status':'inputs_validated','formal_qualification':False,'methods':list(IMPLEMENTED)}))
    else:
        report=run(spec,args.resume);print(json.dumps({key:report[key] for key in ('status','scope','method')}))


if __name__=='__main__':main()
