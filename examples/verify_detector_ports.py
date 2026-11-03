"""Bounded actual-image checks of eight independently implemented policies.

Random initial detectors and small inputs: NOT a performance experiment. Binary
auxiliary labels are an explicit plumbing fixture, not a validated task detector.
"""
from pathlib import Path
import argparse
import copy,gc,hashlib,json,sys,traceback
import torch

from softcas.detector_baselines import new_detector,_load_batch
from softcas.baselines.common import Batches,DetectorTask,model_digest
from softcas.baselines.dpfl import DPFL
from softcas.baselines.dspodfl import DSpodFL
from softcas.baselines.dadpfl import DADPFL
from softcas.baselines.dcfcl import DCFCL
from softcas.baselines.pfeddil import PFedDIL
from softcas.baselines.dflmoe import DFLMoE
from softcas.baselines.demodcl import DeModCL
from softcas.baselines.nas_diffusion import GraphScore,MetaPredictor,GuidedDiffusion,prepare_generator
from softcas.baselines.perfedsde import PerFedSDENAS
from softcas.baselines.runner import source_identity

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--architecture',choices=['YOLOv8n','YOLO11n','YOLO26n'],default='YOLOv8n')
parser.add_argument('--device',required=True)
parser.add_argument('--clients',type=int,choices=[2,4],default=4)
args=parser.parse_args()
work=Path(args.output)
if work.exists() and any(work.iterdir()):raise FileExistsError('Use a new verification output directory')
work.mkdir(parents=True,exist_ok=True)
probe=Path(args.manifest)
manifest=json.loads(probe.read_text(encoding='utf-8'))
if manifest.get('scope')!='native_interface_probe_NOT_FORMAL':raise ValueError('Explicit non-formal probe manifest required')
architecture=args.architecture
count=args.clients
if len(manifest['clients'])<count:raise ValueError('Not enough explicit probe clients')
port_map={'YOLOv8n':[2,4,6,8,15,18],'YOLO11n':[4,6,10,16,19,22],'YOLO26n':[4,6,10,16,19,22]}
torch.set_num_threads(2)
torch.manual_seed(221)
fit=[];select=[]
for client in manifest['clients'][:count]:
 batch=_load_batch(client['fit'],128,args.device);batch['auxiliary_target']=torch.ones(len(batch['img']),device=args.device)
 fit.append(Batches('D',(batch,)))
 select.append(Batches('V',(_load_batch(client['select'],128,args.device),)))

def create(name,models):
 if name=='DPFL':return DPFL(models,learning_rate=1e-4,cardinality=2,pretrain_steps=1,local_steps=1,seed=1)
 if name=='DSpodFL':return DSpodFL(models,adjacency=[[i!=j for j in range(count)] for i in range(count)],gradient_probabilities=[1.]*count,edge_probabilities=[[float(i!=j) for j in range(count)] for i in range(count)],learning_rate=1e-4,seed=1)
 if name=='DA-DPFL':return DADPFL(models,density=.8,target_sparsity=.8,total_rounds=4,max_wait=1,learning_rate=1e-4,local_steps=1,seed=1,rewire_interval=2,early_prune_threshold=100)
 if name=='DCFCL':return DCFCL(models,learning_rate=1e-4,local_steps=1,distillation_weight=.01,temperature=2.,model_similarity_weight=.1)
 if name=='pFedDIL':return PFedDIL(models,threshold=.99,learning_rate=1e-4,local_steps=1)
 if name=='dFLMoE':return DFLMoE(models,learning_rate=1e-4,local_steps=1,moe_steps=1)
 if name=='DeModCL':return DeModCL(models,learning_rate=1e-4,local_steps=1,finetune_steps=1,
  costs=[{'local_seconds':1.,'finetune_seconds':.1,'bandwidth_bytes_per_second':1e9,'metadata_seconds':.001}]*count,
  maximum_task_seconds=100.,storage_bytes=10**9,maximum_loss=1e6,seed=1,
  similarity_threshold=-1.,sharing_threshold=0.,signature_sparsity=1.,signature_perturbation=0.)
 if name=='PerFedSDE-NAS':
  score=GraphScore(7,2,hidden=8,layers=1,heads=2)
  meta=MetaPredictor(2,3,hidden=8)
  adjacency=torch.diag(torch.ones(6),diagonal=1)
  generator=GuidedDiffusion(score,meta,adjacency,sigma_max=.1,steps=2,predictor_std=1.,seed=1)
  arch=torch.nn.functional.one_hot(torch.randint(0,2,(4,7)),2).float()
  masks=torch.ones(4,7,dtype=torch.bool)
  prepare_generator(generator,arch,torch.randn(4,3),torch.tensor([.1,.2,.3,.4]),roles=['SOURCE']*4,paired_architectures=arch.flip(0),paired_masks=(masks,masks),iterations=1,learning_rate=1e-3)
  return PerFedSDENAS(models,port_indices=port_map[architecture],generator=generator,dataset_descriptors=[torch.zeros(3)]*count,learning_rate=1e-4,search_iterations=1,generated_candidates=3,offspring=1,archive_size=2,seed=1,image_size=128)

report={'scope':'bounded_native_policy_check_NOT_FORMAL','architecture':architecture,'input_size':128,'clients':count,'status':'running','checks':{},'source_trained':False,'Orin_measurements':None,'formal_AP':None,'auxiliary_targets':'positive-only plumbing fixture','diffusion_preparation':'four numerical fixture triples, NOT a prepared benchmark model'}
report.update(source_files=source_identity(),torch=torch.__version__,ultralytics='8.4.0',
              device_name=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'CPU',
              DeModCL_costs='numerical admission fixtures, NOT device calibration')
target=work/('native_policies_'+architecture+'.json')
for name in ['DPFL','DSpodFL','DA-DPFL','DCFCL','pFedDIL','dFLMoE','PerFedSDE-NAS','DeModCL']:
 try:
  prototype=new_detector(architecture,args.device,seed=221)
  models=[copy.deepcopy(prototype) for _ in range(count)]
  policy=create(name,models)
  initial=[model_digest(model) for model in policy.models]
  if name=='DPFL':policy.prepare(fit,select)
  if name=='pFedDIL':policy.begin_stage(fit)
  if name=='PerFedSDE-NAS':policy.begin_stage(fit,stage=1)
  logs=[policy.round(fit,select)]
  if name=='pFedDIL':policy.begin_stage(fit)  # Repeat D only for the structural stage-transition check.
  if name=='PerFedSDE-NAS':
   # Explicit repeated-D descriptor fixture: exercise invalidation, not a new scene protocol.
   policy.begin_stage(fit,stage=2,dataset_descriptors=[torch.zeros(3)]*count)
  logs.append(policy.round(fit,select))
  state=copy.deepcopy(policy.state_dict())
  checkpoint=work/(name.replace('/','_')+'_'+architecture+'_probe.pt')
  torch.save(state,checkpoint)
  def predict_client(i):
   images=select[i].values[0]['img']
   return policy.predict(i,images) if hasattr(policy,'predict') else DetectorTask().predictions_from_dense(policy.models[i],DetectorTask().dense(policy.models[i],images))
  with torch.no_grad():before_predictions=[[value.detach().clone() for value in predict_client(i)] for i in range(count)]
  restored=torch.load(checkpoint,map_location='cpu',weights_only=True)
  policy.load_state_dict(restored)
  with torch.no_grad():
   for i in range(count):
    prediction=predict_client(i)
    assert all(torch.isfinite(value).all() for value in prediction)
    for before,after in zip(before_predictions[i],prediction):torch.testing.assert_close(before,after,rtol=1e-5,atol=1e-5)
  assert all(model_digest(model)!=old for model,old in zip(policy.models,initial))
  report['checks'][name]={'status':'passed','rounds':2,'clients_checked':count,'checkpoint_roundtrip':True,'native_decode':True,'reload_predictions_equal':True,'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
  print(architecture,name,'passed',flush=True)
 except Exception as error:
  traceback.print_exc()
  report['checks'][name]={'status':'failed','error':type(error).__name__+': '+str(error)}
  print(architecture,name,'FAILED',str(error),flush=True)
 finally:
  target.write_text(json.dumps(report,indent=2),encoding='utf-8')
  for variable in ['policy','models','prototype','state','restored','prediction']:
   globals().pop(variable,None)
  gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None
report['status']='passed' if all(value['status']=='passed' for value in report['checks'].values()) else 'failed'
target.write_text(json.dumps(report,indent=2),encoding='utf-8')
sys.exit(0 if report['status']=='passed' else 1)
