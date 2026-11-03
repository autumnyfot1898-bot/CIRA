import copy
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from softcas.baseline_algorithms import (average_states, dadpfl_wait_schedule, dcfcl_affinity,
    dpfl_select, dspodfl_step, pq_prune_layer, sparse_average)


def scalar(value):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    model.register_buffer('counter', torch.tensor(7))
    return model


def quadratic(model, batch):
    return model.weight.square().sum() / 2


class BaselineAlgorithmsTest(unittest.TestCase):
    def test_blank_private_endpoint_is_rejected_before_any_connection(self):
        from softcas.transport import TcpClient
        for host in ('', '  ', 'SET_ORIN_ADDRESS', 'BIND_ORIN_0_ADDRESS'):
            with self.assertRaisesRegex(ValueError, 'peer host'):
                TcpClient('0',host,18760,None,'CIRA_PEER_TOKEN')

    def test_dspodfl_atc_and_cta_have_distinct_evaluation_points(self):
        graph = [[False, True], [True, False]]
        for cta, expected in ((False, [1.9, 1.7]), (True, [1.8, 1.8])):
            models = [scalar(1), scalar(3)]
            result = dspodfl_step(models, [None]*2, quadratic, graph, graph, [True]*2, .1, cta)
            self.assertEqual(result['update_order'], 'CTA' if cta else 'ATC')
            for model, value in zip(models, expected):
                self.assertAlmostEqual(model.weight.item(), value, places=6)
                self.assertEqual(model.counter.item(), 7)

    def test_fixed_graph_degrees_and_sporadic_indicators(self):
        graph = [[False, True, True], [True, False, False], [True, False, False]]
        active = [[False, True, False], [True, False, False], [False, False, False]]
        models = [scalar(0), scalar(3), scalar(9)]
        report = dspodfl_step(models, [None]*3, quadratic, graph, active, [False, False, True], .1)
        self.assertAlmostEqual(models[0].weight.item(), 1)
        self.assertAlmostEqual(models[1].weight.item(), 2)
        self.assertAlmostEqual(models[2].weight.item(), 8.1, places=5)
        self.assertEqual(report['local_losses'][:2], [None, None])
        self.assertEqual(report['parameter_tensor_bytes_sent'], 8)

    def test_nonfinite_last_client_does_not_partially_mutate_models(self):
        models = [scalar(1), scalar(2)]
        graph = [[False, True], [True, False]]
        with self.assertRaises(ValueError):
            dspodfl_step(models, [1,float('nan')], lambda m,b: quadratic(m,b)*b,
                          graph, graph, [True,True], .1)
        self.assertEqual([m.weight.item() for m in models], [1,2])

    def test_graph_validation_and_shape_mismatch(self):
        with self.assertRaises(ValueError):
            dspodfl_step([scalar(1), scalar(2)], [None]*2, quadratic,
                          [[False,True],[True,False]], [[False,True],[False,False]], [False]*2, .1)
        with self.assertRaises(ValueError):
            average_states([{'x':torch.ones(2)}, {'x':torch.ones(3)}])

    def test_greedy_retains_owner_and_rejects_harmful_peer(self):
        values = {(0,):1., (0,1):3., (0,2):.5, (0,1,2):2.}
        selected = dpfl_select(0,[0,1,2],3,lambda x:values[tuple(sorted(x))],variant='non_monotone_greedy')
        self.assertEqual(selected.clients,(0,1))
        self.assertGreater(len(selected.oracle_calls),0)

    def test_double_greedy_positive_equal_marginals_and_cardinality(self):
        # f({owner})=f(all)=0, f(owner+a)=f(owner+b)=1 gives equal positive marginals.
        def reward(group):
            return float(len(group)==2)
        class Rng:
            def shuffle(self, values): pass
            def random(self): return .25
        selected = dpfl_select(0,[0,1,2],2,reward,rng=Rng())
        self.assertEqual(selected.clients,(0,1))
        self.assertEqual(len(selected.oracle_calls),4)
        zero = dpfl_select(0,[0,1,2],2,lambda _:0.,rng=Rng())
        self.assertEqual(zero.clients,(0,1))

    def test_subset_nonfinite_and_invalid_availability_are_rejected(self):
        with self.assertRaises(ValueError):
            dpfl_select(0,[0,1],2,lambda _:math.nan,rng=random.Random(1))
        with self.assertRaises(ValueError):
            dpfl_select(0,[0,0],2,lambda _:1,rng=random.Random(1))

    def test_weighted_floating_mean_preserves_local_integer_counter(self):
        a={'w':torch.tensor([1.]),'counter':torch.tensor(3)}
        b={'w':torch.tensor([5.]),'counter':torch.tensor(7)}
        result=average_states([a,b],[1,3],buffer_reference=1)
        self.assertEqual(result['w'].item(),4)
        self.assertEqual(result['counter'].item(),7)
        self.assertEqual(a['w'].item(),1)

    def test_sparse_denominator_uses_active_contributors(self):
        states=[{'w':torch.tensor([2.,0.,6.])},{'w':torch.tensor([4.,8.,0.])}]
        masks=[{'w':torch.tensor([1.,0.,1.])},{'w':torch.tensor([1.,1.,0.])}]
        result=sparse_average(states,masks,{'w':torch.tensor([1.,1.,0.])})
        torch.testing.assert_close(result['w'],torch.tensor([3.,8.,0.]))

    def test_wait_schedule_matches_author_stepwise_rule(self):
        self.assertEqual(dadpfl_wait_schedule(2,8),[2,2,1,1,0,0,0,0])
        with self.assertRaises(ValueError): dadpfl_wait_schedule(3,2)

    def test_pq_pruning_uses_bound_and_retains_magnitude_ties(self):
        w=torch.tensor([.001,.001,.001,5.,6.,7.,8.,9.])
        mask,report=pq_prune_layer(w,torch.ones_like(w),p=.5,q=1.,eta=.1,scale=1.)
        self.assertTrue(torch.all((mask==0)|(mask==1)))
        self.assertGreater(report['pq_index'],0)
        self.assertEqual(mask[0].item(),mask[1].item())
        empty,_=pq_prune_layer(torch.zeros(2),torch.zeros(2),p=.5,q=1.,eta=.1,scale=1.)
        self.assertEqual(empty.sum().item(),0)

    def test_dcfcl_affinity_combines_two_distinct_cosines(self):
        gradients=[torch.tensor([1.,0.]),torch.tensor([-1.,0.])]
        params=[torch.tensor([1.,1.]),torch.tensor([1.,1.])]
        matrix=dcfcl_affinity(gradients,params,.25)
        torch.testing.assert_close(matrix,torch.tensor([[1.25,-.75],[-.75,1.25]]))

    def test_dpfl_native_round_selection_does_not_train_on_selection_batch(self):
        from softcas.detector_baselines import dpfl_round
        models=[scalar(1),scalar(3)]
        roles=[]
        def loss(model,batch):
            roles.append((batch,torch.is_grad_enabled()))
            return (model.weight-batch).square().sum()
        with patch('softcas.detector_baselines.detector_loss',side_effect=loss):
            result=dpfl_round(models,[0.,0.],[10.,10.],learning_rate=.01,cardinality=2,
                              variant='rgl',rng=random.Random(2))
        self.assertEqual(sum(grad for batch,grad in roles if batch==0.),2)
        self.assertTrue(all(not grad for batch,grad in roles if batch==10.))
        self.assertEqual(len(result['selected_clients']),2)

    def test_letterbox_probe_targets_and_image_hash(self):
        from PIL import Image
        from softcas.detector_baselines import _load_batch
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            image=Path(directory)/'image.png'
            Image.new('RGB',(100,50)).save(image)
            record={'image':str(image),'sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                    'boxes':[[0,.5,.5,1.,1.]]}
            batch=_load_batch([record],100,'cpu')
            torch.testing.assert_close(batch['bboxes'],torch.tensor([[.5,.5,1.,.5]]))
            record['sha256']='0'*64
            with self.assertRaises(ValueError):_load_batch([record],100,'cpu')


if __name__=='__main__': unittest.main()
