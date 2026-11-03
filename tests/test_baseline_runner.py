import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from PIL import Image

from softcas.baselines.runner import ImageBatches,split,validate_spec,source_identity,run
from softcas.baselines.factory import IMPLEMENTED,bound_torch_asset,create
from softcas.baselines.common import Batches,DetectorTask
from softcas.baselines.nas_diffusion import SetDatasetEncoder


class BaselineRunnerTests(unittest.TestCase):
    def test_eight_algorithms_are_distinct_implementations(self):
        self.assertEqual(len(IMPLEMENTED),8)
        self.assertEqual(len(set(IMPLEMENTED.values())),8)
        self.assertTrue(all(cls.__module__.startswith('softcas.baselines.') for cls in IMPLEMENTED.values()))
        self.assertNotIn('softcas',IMPLEMENTED)
        with self.assertRaises(ValueError):create('softcas',[],{})

    def test_private_template_fails_before_any_model_or_network_access(self):
        path=Path(__file__).parents[1]/'configs/detector_baseline.example.json'
        with patch('softcas.baselines.runner.new_detector') as model:
            with self.assertRaisesRegex(ValueError,'Fill device'):validate_spec(json.loads(path.read_text(encoding='utf-8')))
            model.assert_not_called()

    def test_all_method_templates_preserve_blank_personal_paths(self):
        root=Path(__file__).parents[1]/'configs/detector_baselines'
        documents=[json.loads(path.read_text(encoding='utf-8')) for path in root.glob('*.json')]
        self.assertEqual({doc['method'] for doc in documents},set(IMPLEMENTED))
        for doc in documents:
            self.assertEqual(doc['device'],'');self.assertEqual(doc['output'],'')
            self.assertEqual(doc['checkpoint']['path'],'');self.assertEqual(doc['registry']['path'],'')
            self.assertFalse(doc['formal'])

    def test_checked_tensor_asset_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'asset.pt';torch.save({'weight':torch.ones(2)},path)
            binding={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(bound_torch_asset(binding)['weight'].sum().item(),2)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):bound_torch_asset(binding)

    def test_image_batch_preserves_binary_membership_and_role_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'image.png';Image.new('RGB',(80,40)).save(path)
            base={'image':str(path),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'boxes':[[2,.5,.5,.5,.5]],'client':'0','stage':1,'domain':'fog','auxiliary_target':1}
            records=[dict(base,role='D'),dict(base,role='T')]
            data=split(records,'0',1,'D',directory,64,1,'cpu')
            self.assertEqual(data.role,'D');self.assertEqual(len(data.values),1)
            batch=data.values[0]
            self.assertEqual(batch['auxiliary_target'].item(),1)
            torch.testing.assert_close(batch['bboxes'],torch.tensor([[.5,.5,.5,.25]]))
            test=split(records,'0',1,'T',directory,64,1,'cpu','fog')
            with self.assertRaises(ValueError):test.require('D')

    def test_set_dataset_encoder_masks_absent_classes_without_NaNs(self):
        encoder=SetDatasetEncoder(3,8)
        values=torch.randn(2,4,3,3)
        valid=torch.ones(2,4,3,dtype=torch.bool);valid[:,2]=False
        result=encoder({'values':values,'valid':valid})
        self.assertEqual(result.shape,(2,8));self.assertTrue(torch.isfinite(result).all())
        altered=values.clone();altered[:,2]=9999
        torch.testing.assert_close(result,encoder({'values':altered,'valid':valid}))
        with self.assertRaises(ValueError):encoder({'values':values,'valid':torch.zeros_like(valid)})

    def test_source_identity_covers_common_loss_and_all_algorithms(self):
        names=source_identity()
        self.assertIn('evaluation.py',names);self.assertIn('baseline_algorithms.py',names)
        self.assertIn('baselines/demodcl.py',names);self.assertIn('baselines/runner.py',names)

    def test_multistage_protocol_roles_client_artifacts_and_completed_resume(self):
        # This mocked-model test checks protocol routing only; its readout is
        # not included in native evidence and never supplies paper AP numbers.
        observed=[]
        class Policy:
            def __init__(self,models):self.models=models;self.count=0
            def round(self,data,selection,available):
                for model,stream in zip(self.models,data):
                    self.assert_D=stream.require('D')
                    observed.append(stream.role)
                    with torch.no_grad():model.weight.add_(.01)
                self.count+=1;return {'count':self.count}
            def state_dict(self):return {'models':[m.state_dict() for m in self.models],'count':self.count}
            def load_state_dict(self,state):
                for model,saved in zip(self.models,state['models']):model.load_state_dict(saved)
                self.count=state['count']
        class Task:
            def ap(self,model,data,role='V',predict=None):
                data.require(role);observed.append(role);return .5
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);records=[];color=0
            for stage,domain in [(1,'clear'),(2,'fog')]:
                for client in ['0','1']:
                    for role in ['D','V','A']:
                        path=root/(str(color)+'.png');Image.new('RGB',(64,32),(color,0,0)).save(path)
                        records.append({'image':str(path),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'scene_id':'scene'+str(color),'role':role,'domain':domain,'client':client,'stage':stage,
                            'boxes':[[2,.5,.5,.4,.4]]});color+=1
                path=root/(str(color)+'.png');Image.new('RGB',(64,32),(color,0,0)).save(path)
                records.append({'image':str(path),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'scene_id':'scene'+str(color),'role':'T','domain':domain,'client':'shared','stage':0,
                    'boxes':[[2,.5,.5,.4,.4]]});color+=1
            from softcas.detector_baselines import LABELS
            registry=root/'registry.json';registry.write_text(json.dumps({'labels':LABELS,'records':records}),encoding='utf-8')
            checkpoint=root/'base.pt';torch.save({'fixture':'protocol_only_NOT_AP'},checkpoint)
            binding=lambda p:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            spec={'schema':'cira.detector_port_run.v1','scope':'NATIVE_ADAPTATION_VERIFICATION','formal':False,
                'method':'DSpodFL','detector':'YOLOv8n','dataset':'fixture','seed':1,'device':'cpu','image_size':64,
                'checkpoint':binding(checkpoint),'registry':binding(registry),'output':str(root/'run'),
                'clients':['0','1'],'stages':[1,2],'communication_iterations_per_stage':1,
                'fit_batch_size':1,'evaluation_batch_size':1,'options':{},'meter':{'strict':False}}
            with patch('softcas.baselines.runner.new_detector',side_effect=lambda *a,**k:nn.Linear(2,2,bias=False)),\
                 patch('softcas.baselines.runner.create',side_effect=lambda name,models,options,assets:Policy(models)),\
                 patch('softcas.baselines.runner.DetectorTask',Task):
                result=run(spec);self.assertEqual(result['status'],'complete_native_verification')
                self.assertFalse(result['formal_experiment_complete']);self.assertFalse(result['network_accounting_complete'])
                matrix=json.loads((root/'run/matrix.json').read_bytes())
                self.assertEqual(len(matrix),6)
                self.assertNotEqual(matrix[0]['deployed'],matrix[1]['deployed'])
                before=len(observed);run(spec,resume=True)
                self.assertEqual(len(observed),before)
            self.assertEqual(observed.count('D'),4);self.assertEqual(observed.count('T'),6)
            self.assertEqual(len((root/'run/attempts.jsonl').read_text().splitlines()),2)


if __name__=='__main__':unittest.main()
