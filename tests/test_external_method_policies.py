import copy
import random
import unittest

import torch
from torch import nn

from softcas.baselines.common import Batches, snapshot, model_digest
from softcas.baselines.dpfl import DPFL, batched_graph
from softcas.baselines.dspodfl import DSpodFL
from softcas.baselines.dadpfl import DADPFL, layer_densities, rewire_mask
from softcas.baselines.dcfcl import DCFCL, benefit_table, merge_blocking, blocks, coalitions
from softcas.baselines.pfeddil import PFedDIL
from softcas.baselines.demodcl import DeModCL,MemoryTrace,AnchorMapping,sharing_score
from softcas.baselines.nas_diffusion import GraphScore,MetaPredictor,GuidedDiffusion,prepare_generator
from softcas.baselines.perfedsde import RBF,nonzero_aggregate,PerFedSDENAS


class Linear(nn.Module):
    def __init__(self,value=1.):
        super().__init__();self.layer=nn.Linear(2,2,bias=False)
        with torch.no_grad():self.layer.weight.fill_(value)
    def forward(self,value):return self.layer(value)


def objective(model,batch):return (model(batch['img'])-batch['target']).square().mean()


def streams(role='D'):
    return Batches(role,({'img':torch.tensor([[1.,0.],[0.,1.]]),'target':torch.zeros(2,2),
                          'auxiliary_target':torch.tensor([0.,1.])},))


class SmallTask:
    def features(self,model,image):return model(image)
    def loss_features(self,model,batch):
        value=model(batch['img']);return (value-batch['target']).square().mean(),value
    def dense(self,model,image):return model(image)[:,:,None]
    def predictions_from_dense(self,model,dense):return dense


class ExternalPolicyTests(unittest.TestCase):
    def test_role_check_blocks_T_fitting(self):
        with self.assertRaises(ValueError):streams('T').require('D')

    def test_bgcc_two_pass_stream_and_cardinality(self):
        states=[{'w':torch.tensor([float(i)])} for i in range(4)]
        selected,log=batched_graph(0,[1,2,3],states[0],[1]*4,2,lambda i:states[i],
                                  lambda state:float(state['w']),random.Random(1))
        self.assertEqual(selected[0],0);self.assertLessEqual(len(selected),2)
        self.assertEqual(log['fetches'][:3],[1,2,3])
        self.assertLessEqual(log['peak_peer_batch'],2)

    def test_dpfl_graph_survives_round_and_checkpoint(self):
        options=dict(learning_rate=.01,cardinality=2,pretrain_steps=1,local_steps=1,seed=1,loss_fn=objective)
        method=DPFL([Linear(1),Linear(2)],**options)
        prepared=method.prepare([streams()]*2,[streams('V')]*2)
        graph=copy.deepcopy(method.graph)
        method.round([streams()]*2,[streams('V')]*2)
        self.assertEqual(method.graph,graph)
        resumed=DPFL([Linear(),Linear()],**options);resumed.load_state_dict(method.state_dict())
        first=method.round([streams()]*2,[streams('V')]*2)
        second=resumed.round([streams()]*2,[streams('V')]*2)
        self.assertEqual(first['selected_clients'],second['selected_clients'])
        self.assertEqual([model_digest(m) for m in method.models],[model_digest(m) for m in resumed.models])
        self.assertGreater(prepared['selection_E'],0)

    def test_dspodfl_event_rng_resumes_exactly(self):
        options=dict(adjacency=[[False,True],[True,False]],gradient_probabilities=[.5,.7],
            edge_probabilities=[[0.,.6],[.6,0.]],learning_rate=.1,seed=1,loss_fn=objective)
        method=DSpodFL([Linear(1),Linear(3)],**options)
        method.round([streams()]*2)
        resumed=DSpodFL([Linear(),Linear()],**options);resumed.load_state_dict(method.state_dict())
        self.assertEqual(method.round([streams()]*2),resumed.round([streams()]*2))
        self.assertEqual([model_digest(m) for m in method.models],[model_digest(m) for m in resumed.models])

    def test_sparse_ERK_budget_and_regrowth_count(self):
        parameters={'a':torch.ones(2,2),'b':torch.ones(12,12)}
        densities=layer_densities(parameters,.4)
        actual=sum(parameters[k].numel()*value for k,value in densities.items())
        self.assertAlmostEqual(actual,.4*sum(v.numel() for v in parameters.values()))
        w=torch.arange(8.).reshape(2,4);mask=torch.tensor([[1.,1.,1.,1.],[0.,0.,0.,0.]])
        result,count=rewire_mask(w,mask,torch.arange(8.).reshape(2,4),.5)
        self.assertEqual(count,2);self.assertEqual(result.sum().item(),mask.sum().item())
        self.assertTrue(result.flatten()[-1])

    def test_dadpfl_sparse_schedule_checkpoint_and_full_D_gradient(self):
        options=dict(density=.5,target_sparsity=.9,total_rounds=5,max_wait=1,
            learning_rate=.01,local_steps=1,seed=1,distribution='uniform',rewire_interval=2,
            early_prune_threshold=100,loss_fn=objective)
        method=DADPFL([Linear(1),Linear(2)],**options)
        method.round([streams()]*2);report=method.round([streams()]*2)
        self.assertEqual(report['trigger_round'],1)
        resumed=DADPFL([Linear(),Linear()],**options);resumed.load_state_dict(method.state_dict())
        method.round([streams()]*2);resumed.round([streams()]*2)
        self.assertEqual([model_digest(m) for m in method.models],[model_digest(m) for m in resumed.models])
        for model,masks in zip(method.models,method.masks):
            for name,p in model.named_parameters():
                if name in masks:self.assertTrue(torch.all(p.detach()[masks[name]==0]==0))

    def test_DCFCL_benefits_keep_two_vector_families_separate(self):
        gradients=[torch.tensor([1.,0.]),torch.tensor([1.,0.]),torch.tensor([-1.,0.]),torch.tensor([-1.,0.])]
        parameters=[torch.tensor([1.,1.])]*4
        table=benefit_table(gradients,parameters,[1.]*4,.2)
        self.assertAlmostEqual(table[(0,1)][0],1.2)
        partition,trace=merge_blocking([(i,) for i in range(4)],table)
        self.assertEqual(partition,((0,1),(2,3)))
        self.assertTrue(trace)
        self.assertFalse(any(blocks(group,partition,table) for group in coalitions(range(4))))

    def test_DCFCL_actual_round_and_resume(self):
        options=dict(learning_rate=.01,local_steps=1,distillation_weight=0.,temperature=2.,
                     model_similarity_weight=.1,loss_fn=objective)
        method=DCFCL([Linear(1),Linear(2)],**options)
        method.round([streams()]*2)
        resumed=DCFCL([Linear(),Linear()],**options);resumed.load_state_dict(method.state_dict())
        self.assertEqual(method.round([streams()]*2)['partition'],resumed.round([streams()]*2)['partition'])
        self.assertEqual([model_digest(m) for m in method.models],[model_digest(m) for m in resumed.models])

    def test_MBA_certificate_does_not_hide_non_equilibrium_or_replace_algorithm(self):
        torch.manual_seed(14)
        table=benefit_table([torch.randn(10) for _ in range(4)],[torch.randn(10) for _ in range(4)],[1.]*4,.2)
        partition,trace=merge_blocking([(i,) for i in range(4)],table)
        self.assertFalse(trace[-1]['equilibrium_verified'])
        self.assertTrue(trace[-1]['remaining_blocking_coalitions'])

    def test_DeModCL_trace_statistics_coverage_and_anchor_order(self):
        trace=MemoryTrace(2,capacity=3)
        values=torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
        trace.update(values)
        torch.testing.assert_close(trace.mean,values.mean(0))
        torch.testing.assert_close(trace.variance,values.var(0,unbiased=False))
        self.assertEqual(trace.n,3)
        self.assertAlmostEqual(trace.feasibility(values,beta_d=1.,gamma1=.4,coverage_tolerance=1.),1.,places=5)
        mapping=AnchorMapping(2,8,4,seed=1)
        torch.testing.assert_close(mapping(values),mapping(values.flip(0)))
        self.assertAlmostEqual(sharing_score(.8,10.,1.,1.,.5),.9,places=5)

    def test_DeModCL_new_share_reuse_no_eviction_and_checkpoint(self):
        options=dict(learning_rate=.001,local_steps=1,finetune_steps=1,costs=[
            {'local_seconds':1.,'finetune_seconds':.1,'bandwidth_bytes_per_second':1e9,'metadata_seconds':.001}]*2,
            maximum_task_seconds=10.,storage_bytes=10**8,maximum_loss=10**6,seed=1,
            similarity_threshold=-1.,sharing_threshold=0.,signature_sparsity=1.,signature_perturbation=0.,
            anchor_dimension=8,task=SmallTask())
        method=DeModCL([Linear(1),Linear(1)],**options)
        result=method.round([streams()]*2,now=1.)
        self.assertEqual(result['clients'][0]['decision'],'new')
        self.assertEqual(result['clients'][1]['decision'],'share')
        self.assertTrue(result['clients'][1]['signature_published'])
        result=method.round([streams()]*2,now=2.)
        self.assertEqual(result['clients'][0]['decision'],'reuse')
        resumed=DeModCL([Linear(1),Linear(1)],**options);resumed.load_state_dict(method.state_dict())
        torch.testing.assert_close(method.predict(0,streams().values[0]['img']),resumed.predict(0,streams().values[0]['img']))
        old_size=len(method.pools[0]);method.capacity=1
        result=method.round([streams()]*2,now=3.)
        self.assertTrue(result['clients'][0]['status'].startswith('infeasible'))
        self.assertEqual(len(method.pools[0]),old_size)

    def test_DeModCL_C3_suppresses_signature_and_stale_cache(self):
        costs=[{'local_seconds':1.,'finetune_seconds':.1,'bandwidth_bytes_per_second':1e9,'metadata_seconds':.001}]*2
        method=DeModCL([Linear(1),Linear(1)],learning_rate=.001,local_steps=1,finetune_steps=1,costs=costs,
            maximum_task_seconds=10.,storage_bytes=10**8,maximum_loss=1e-12,seed=1,anchor_dimension=8,task=SmallTask())
        report=method.round([streams()]*2,now=1.)
        self.assertTrue(all(row['status']=='infeasible_C3' for row in report['clients']))
        self.assertEqual(method.caches,[{},{}]);self.assertEqual(method.pools,[[],[]])

    def test_pFedDIL_portfolio_matching_sharing_and_resume(self):
        options=dict(threshold=.99,learning_rate=.01,local_steps=1,task=SmallTask())
        method=PFedDIL([Linear(1),Linear(2)],**options)
        decision=method.begin_stage([streams()]*2)
        self.assertTrue(all(x['decision']=='new' for x in decision['matching']))
        method.round([streams()]*2)
        method.begin_stage([streams()]*2)
        self.assertEqual([len(p.models) for p in method.portfolios],[2,2])
        method.round([streams()]*2)
        method.portfolios[0].threshold=0.
        self.assertEqual(method.portfolios[0].begin_stage(streams(),method.global_model)['decision'],'inherit')
        resumed=PFedDIL([Linear(),Linear()],**options);resumed.load_state_dict(method.state_dict())
        torch.testing.assert_close(method.predict(1,streams().values[0]['img']),resumed.predict(1,streams().values[0]['img']))

    def test_auxiliary_targets_cannot_be_omitted_or_inferred_from_detection_classes(self):
        method=PFedDIL([Linear()],threshold=.5,learning_rate=.01,local_steps=1,task=SmallTask())
        bad=dict(streams().values[0]);del bad['auxiliary_target']
        data=Batches('D',(bad,));method.begin_stage([data])
        with self.assertRaisesRegex(ValueError,'auxiliary_target'):method.round([data])

    def test_nonzero_aggregation_uses_per_coordinate_denominator(self):
        base={'w':torch.tensor([1.,2.,3.]),'counter':torch.tensor(9)}
        clients=[{'w':torch.tensor([2.,2.,3.]),'counter':torch.tensor(1)},
                 {'w':torch.tensor([1.,4.,3.]),'counter':torch.tensor(2)}]
        result=nonzero_aggregate(base,clients,[1.,3.])
        torch.testing.assert_close(result['w'],torch.tensor([2.,4.,3.]))
        self.assertEqual(result['counter'].item(),9)

    def test_NAS_new_domain_requires_current_D_descriptor_and_clears_old_fitness(self):
        method=object.__new__(PerFedSDENAS)
        method.models=[Linear(),Linear()]
        method.stage=0
        method.descriptors=[torch.zeros(3),torch.ones(3)]
        method.descriptors_by_stage={'2':[torch.full((3,),2.),torch.full((3,),3.)]}
        method.population=[{(1,0):.7},{(0,1):.8}]
        method.begin_stage([streams()]*2,stage=1)
        self.assertEqual(method.population,[{},{}])
        method.population=[{(1,0):.5},{(0,1):.6}]
        method.begin_stage([streams()]*2,stage=2)
        torch.testing.assert_close(method.descriptors[0],torch.full((3,),2.))
        self.assertEqual(method.population,[{},{}])
        with self.assertRaisesRegex(ValueError,'current-stage'):
            method.begin_stage([streams()]*2,stage=3)
        self.assertEqual(method.stage,2)
        with self.assertRaises(ValueError):method.begin_stage([streams('T')]*2,stage=3)

    def test_RBF_interpolation_and_conditional_diffusion_have_real_gradients(self):
        predictor=RBF([(0,0),(1,1),(1,0)],[0.,1.,.4])
        torch.testing.assert_close(predictor.predict([(0,0),(1,1),(1,0)]),torch.tensor([0.,1.,.4],dtype=torch.float64),atol=2e-6,rtol=1e-5)
        torch.manual_seed(1)
        score=GraphScore(7,2,hidden=8,layers=1,heads=2)
        meta=MetaPredictor(2,3,hidden=8)
        adjacency=torch.diag(torch.ones(6),diagonal=1)
        generator=GuidedDiffusion(score,meta,adjacency,steps=2,sigma_max=.1,predictor_std=1.,seed=1)
        arch=torch.nn.functional.one_hot(torch.randint(0,2,(4,7)),2).float()
        masks=torch.ones(4,7,dtype=torch.bool)
        report=prepare_generator(generator,arch,torch.randn(4,3),torch.tensor([.1,.2,.3,.4]),roles=['SOURCE']*4,
                                 paired_architectures=arch.flip(0),paired_masks=(masks,masks),iterations=1,learning_rate=1e-3)
        self.assertTrue(generator.prepared)
        architectures=generator.generate(torch.zeros(3),3)
        self.assertEqual(len(architectures),3);self.assertTrue(all(any(a[:6]) for a in architectures))
        self.assertEqual(report['preparation_examples'],4)
        with self.assertRaises(ValueError):prepare_generator(generator,arch,torch.randn(4,3),torch.zeros(4),roles=['T']*4,
            paired_architectures=arch,paired_masks=(masks,masks),iterations=1,learning_rate=1e-3)


if __name__=='__main__':unittest.main()
