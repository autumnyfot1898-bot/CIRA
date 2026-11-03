"""Regression tests for proposal admission and explicit native R1000 rules."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch

from softcas.actions import Proposal, TYPES
from softcas.backend import Backend
from softcas.cache import ContactExchange, PackageCache
from softcas.config import load_config
from softcas.controller import Archive, FeatureEncoder, Ridge, Scored, Selector
from softcas.engine import initial_state
from softcas.model import Adapter
from softcas.proposals import ProposalFactory
from softcas.resources import Cost, Ledger
from softcas.state import FitSettings, Package, Structure, canonical, freeze_tensors

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class ProposalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT/'configs/fixture.json')
        self.backend = Backend()
        self.cache = PackageCache(1000000, self.backend.specs)
        self.state = initial_state(self.backend, self.cache, Structure((1,0,0,0,0,0), 1/8), FitSettings(1e-3,1e-3))
        self.ledger = Ledger(Cost(1e12,100,1e7,1000), Cost())
        self.exchange = ContactExchange('client0', self.cache, self.ledger)

    def native(self):
        self.config['proposal_protocol'] = 'native_r1000'
        self.config['randomization'] = dict(dataset='ControlledFog',detector='test_detector',client='client0',stage=2)
        return self.config

    def archive(self, state=None):
        return Archive(state or self.state, 0.5, 8, self.cache)

    def package(self, port=0, ratio=1/8, version='1'):
        adapter = Adapter(self.backend.specs[port].channels, ratio)
        package = Package('peer', version, 0, 0, self.backend.specs[port], ratio,
                          freeze_tensors(adapter.state_dict()), 'UNIT_TEST_ONLY')
        self.cache.put(package.hash, package.wire)
        return package

    def test_named_streams_are_replayable_and_independent(self):
        factory = ProposalFactory(self.native(), self.backend, self.exchange)
        a = factory._stream('variation.crossover', 2).integers(1000000, size=10)
        factory._stream('unrelated.operation', 9).integers(1000000, size=400)
        b = factory._stream('variation.crossover', 2).integers(1000000, size=10)
        self.assertEqual(a.tolist(), b.tolist())
        row = factory.randomization_records[0]
        expected = int.from_bytes(hashlib.sha256(canonical(row['seed_tuple'])).digest()[:8], 'big')
        self.assertEqual(row['seed'], expected)
        self.assertEqual(row['generator'], 'numpy.PCG64')
        self.assertEqual(row['seed_tuple'][:5], (1001,'ControlledFog','test_detector','client0',2))

    def test_native_replays_across_archive_insertion_order(self):
        c = self.native()
        x, y = self.state.lineage(id=1), self.state.lineage(id=2)
        a, b = self.archive(), self.archive()
        for archive, states in [(a,[x,y]),(b,[y,x])]:
            for state in states:
                archive.insert(Scored(state,0.6,0.1,'configuration'))
        left = ProposalFactory(copy.deepcopy(c),self.backend,self.exchange).generate(a,self.ledger)
        right = ProposalFactory(copy.deepcopy(c),self.backend,self.exchange).generate(b,self.ledger)
        self.assertEqual([p.hash for p in left], [p.hash for p in right])
        self.assertTrue(left)
        self.assertTrue(all(sum(p.type==t for p in left)<=3 for t in TYPES))
        self.assertTrue(all(p.z is None or 1<=sum(p.z.mask)<=2 for p in left))

    def test_native_variation_preserves_fixed_gene_families(self):
        factory = ProposalFactory(self.native(),self.backend,self.exchange)
        other = replace(self.state,z=Structure((0,1,1,0,0,0),1/16),h=FitSettings(1e-4,1e-4))
        for slot in range(20):
            z,h = factory._native_variation(self.state,other,slot,fix_z=True)
            self.assertEqual(z,self.state.z)
            z,h = factory._native_variation(self.state,other,slot,fix_h=True)
            self.assertEqual(h,self.state.h)
            self.assertIn(sum(z.mask),(1,2))

    def test_affordable_candidates_are_filtered_before_bounded_draw(self):
        self.config.update(method='local_nas_hpo',proposals_per_type=1)
        factory = ProposalFactory(self.config,self.backend,self.exchange)
        def estimate(kind,*args,**kwargs):
            return Cost(F=2e12 if kind=='fresh' else 1,E=1)
        with patch.object(factory,'estimate',side_effect=estimate):
            proposals = factory.generate(self.archive(),self.ledger)
        self.assertEqual([p.kind for p in proposals],['refine'])

    def test_native_duplicate_identity_is_merged_without_filling_short_pool(self):
        factory = ProposalFactory(self.native(),self.backend,self.exchange)
        with patch.object(factory,'_stream',side_effect=lambda *args:np.random.default_rng(9)):
            proposals = factory.generate(self.archive(),self.ledger)
        self.assertEqual(len(proposals),2)  # one Fresh and one Refine, not three copied entries
        a = proposals[0]
        self.assertEqual(factory._identity(a),factory._identity(replace(a,estimate=Cost(F=99))))
        self.assertNotEqual(factory._identity(a),factory._identity(replace(a,seed=a.seed+1)))

    def test_remove_final_member_has_zero_receiving_fidelity(self):
        package = self.package()
        parent = replace(self.state,C=(package.hash,),beta=((package.hash,0.0),))
        self.config['method']='fixed_collaboration'
        proposals = ProposalFactory(self.config,self.backend,self.exchange).generate(self.archive(parent),self.ledger)
        removals = [p for p in proposals if p.kind=='remove']
        self.assertTrue(removals)
        for p in removals:
            self.assertEqual(p.fidelity,0)
            self.assertEqual(p.estimate.F,self.config['estimates']['evaluation_F'])

    def test_remaining_receiving_parameters_still_fit_after_remove(self):
        p,q = self.package(),self.package(port=1)
        members = tuple(sorted([p.hash,q.hash]))
        parent = replace(self.state,C=members,beta=tuple((key,0.0) for key in members))
        self.config['method']='fixed_collaboration'
        proposals = ProposalFactory(self.config,self.backend,self.exchange).generate(self.archive(parent),self.ledger)
        removals = [p for p in proposals if p.kind=='remove']
        self.assertTrue(removals)
        self.assertTrue(all(p.fidelity==self.config['beta_steps'] for p in removals))

    def test_discrete_replace_estimate_removes_old_exact_member(self):
        old,new = self.package(ratio=1/16),self.package(ratio=1/8,version='2')
        parent = replace(self.state,C=(old.hash,),beta=((old.hash,0.0),))
        self.config.update(method='fixed_collaboration',ablation='discrete')
        calls=[]
        class Profile:
            def estimate(_self,signature,steps,acquisition_bytes,descriptors):
                calls.append((signature,steps,descriptors))
                return Cost(F=1,E=1)
        self.backend.cost_profile=Profile()
        ProposalFactory(self.config,self.backend,self.exchange).generate(self.archive(parent),self.ledger)
        signature = self.backend.signature_parts(parent.z,[(0,new.ratio)],'discrete','collaboration')
        replacements=[r for r in calls if r[0]==signature]
        self.assertTrue(replacements)
        for _,steps,desc in replacements:
            self.assertEqual(steps,0)
            self.assertEqual([b for b in desc['branches'] if b[2]=='external'],[(0,new.ratio,'external')])

    def test_import_cost_uses_complete_target_not_reference_parent(self):
        target=replace(self.state,z=Structure((0,0,0,0,1,0),1/8)).lineage(id='retained')
        class Profile:
            def estimate(_self,signature,steps,acquisition_bytes,descriptors):
                return Cost(F=sum(port+1 for port,ratio,owner in descriptors['branches']),E=1)
        self.backend.cost_profile=Profile()
        self.config['method']='fixed_collaboration'
        proposals=ProposalFactory(self.config,self.backend,self.exchange,[target]).generate(self.archive(),self.ledger)
        imports=[p for p in proposals if p.kind=='import']
        self.assertTrue(imports)
        self.assertTrue(all(p.parent==self.state.hash and p.estimate.F==5 and p.fidelity==0 for p in imports))

    def test_epoch_fidelity_is_clipped_before_estimation(self):
        self.native()
        self.config.update(fitting=dict(unit='epochs',stage_epochs=500,local_epochs=32,receiving_epochs=16),
                           _fit_batches_per_epoch=3,_fit_steps_remaining=7)
        proposals=ProposalFactory(self.config,self.backend,self.exchange).generate(self.archive(),self.ledger)
        self.assertTrue(proposals)
        for p in proposals:
            self.assertEqual(p.fidelity,7)
            self.assertEqual(p.estimate.F,self.config['estimates']['evaluation_F']+7*self.config['estimates']['step_F'])

    def test_serial_epoch_quota_clips_first_phase(self):
        self.native()
        self.config.update(method='serial',fitting=dict(unit='epochs',stage_epochs=10,local_epochs=32,receiving_epochs=16),
                           _fit_batches_per_epoch=3,_fit_steps_remaining=18)
        proposals=ProposalFactory(self.config,self.backend,self.exchange).generate(self.archive(),self.ledger)
        self.assertTrue(proposals)
        self.assertTrue(all(p.fidelity==3 for p in proposals))

    def test_serial_epoch_phase_is_not_split_by_evaluation_or_compute_fraction(self):
        self.native()
        self.config.update(method='serial',fitting=dict(unit='epochs',stage_epochs=10,local_epochs=32,receiving_epochs=16),
                           _fit_batches_per_epoch=3,_fit_steps_remaining=30)
        self.ledger.charge('already_accounted',Cost(F=6e11,E=60))
        factory=ProposalFactory(self.config,self.backend,self.exchange)
        proposals=factory.generate(self.archive(),self.ledger)
        self.assertEqual(factory.phase,0)
        self.assertTrue(proposals)
        self.assertTrue(all(p.type=='configuration' and p.fidelity==15 for p in proposals))

    def test_serial_second_phase_cannot_reassign_unused_first_epoch_quota(self):
        self.native()
        package=self.package()
        parent=replace(self.state,C=(package.hash,),beta=((package.hash,0.0),))
        self.package(port=1)
        self.config.update(method='serial',fitting=dict(unit='epochs',stage_epochs=10,local_epochs=32,receiving_epochs=16),
                           _fit_batches_per_epoch=3,_fit_steps_remaining=24)
        archive=self.archive(parent)
        factory=ProposalFactory(self.config,self.backend,self.exchange)
        factory.advance_serial(archive,self.ledger)
        proposals=factory.generate(archive,self.ledger)
        fitting=[p for p in proposals if p.fidelity]
        self.assertTrue(fitting)
        self.assertTrue(all(p.fidelity==15 for p in fitting))
        self.config['_fit_steps_remaining']=9  # second phase has spent its 15 allocated steps
        later=factory.generate(archive,self.ledger)
        self.assertTrue(later)
        self.assertTrue(all(p.kind=='remove' and p.fidelity==0 for p in later))

    def test_epoch_exhaustion_keeps_only_inherently_zero_fit_edits(self):
        self.native()
        package=self.package()
        self.package(port=1)
        parent=replace(self.state,C=(package.hash,),beta=((package.hash,0.0),))
        self.config.update(fitting=dict(unit='epochs',stage_epochs=10,local_epochs=32,receiving_epochs=16),
                           _fit_batches_per_epoch=3,_fit_steps_remaining=0)
        factory=ProposalFactory(self.config,self.backend,self.exchange,[self.state])
        proposals=factory.generate(self.archive(parent),self.ledger)
        self.assertTrue(proposals)
        self.assertEqual({p.kind for p in proposals},{'remove','import'})
        self.assertTrue(all(p.fidelity==0 for p in proposals))

    def test_epoch_exhaustion_filters_remove_when_receiving_parameters_remain(self):
        self.native()
        p,q=self.package(),self.package(port=1)
        members=tuple(sorted([p.hash,q.hash]))
        parent=replace(self.state,C=members,beta=tuple((key,0.0) for key in members))
        self.config.update(fitting=dict(unit='epochs',stage_epochs=10,local_epochs=32,receiving_epochs=16),
                           _fit_batches_per_epoch=3,_fit_steps_remaining=0)
        proposals=ProposalFactory(self.config,self.backend,self.exchange).generate(self.archive(parent),self.ledger)
        self.assertFalse(proposals)

    def test_cached_add_features_use_verified_package_metadata(self):
        package=self.package()
        a=Proposal(self.state.hash,'add',1,Cost(),add=package.hash)
        encoder=FeatureEncoder(['package_bytes','package_missing'],dict(package_bytes=1,package_missing=1))
        features=encoder.encode(a,Scored(self.state,0.5,0,'incumbent'),self.ledger.remaining(),self.cache,{})
        self.assertEqual(features,(float(len(package.wire)),0.0))
        with self.assertRaises(ValueError):
            encoder.encode(replace(a,add='unseen'),Scored(self.state,0.5,0,'incumbent'),self.ledger.remaining(),self.cache,{})

    def test_invalid_feature_scales_and_ridge_predictions_fail_closed(self):
        for scale in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):FeatureEncoder(['bias'],dict(bias=scale))
        with self.assertRaises(ValueError):Ridge(25,float('nan'))
        ridge=Ridge(25,1)
        for feature in [[0]*24,[float('nan')]*25]:
            with self.assertRaises(ValueError):ridge.predict(feature)

    def test_native_cold_start_does_not_revisit_skipped_local_offer_or_type(self):
        package=self.package()
        parent=replace(self.state,C=(package.hash,),beta=((package.hash,0.0),))
        archive=self.archive(parent)
        archive.insert(Scored(self.state,0.5,0,'configuration'))
        encoder=FeatureEncoder(**self.config['features'])
        selector=Selector(TYPES,TYPES,0,0.001,proposal_protocol='native_r1000')
        ridge=Ridge(len(encoder.fields),1)
        remove=Proposal(parent.hash,'remove',1,Cost(E=1),remove=package.hash)
        decision,reason=selector.choose([remove],archive,ridge,encoder,self.ledger.remaining(),self.cache,{})
        self.assertEqual(reason,'cold_type')
        selector.observed.add('cached')
        fresh=Proposal(self.state.hash,'fresh',2,Cost(E=1),z=self.state.z,h=self.state.h)
        decision,reason=selector.choose([fresh],archive,ridge,encoder,self.ledger.remaining(),self.cache,{})
        self.assertIsNone(decision)
        self.assertEqual(reason,'keep')
        self.assertTrue(selector.local_fresh_offered)

    def test_duplicate_type_order_is_rejected(self):
        with self.assertRaises(ValueError):Selector(TYPES+('configuration',),TYPES,0,0.001)


if __name__ == '__main__':
    unittest.main()
