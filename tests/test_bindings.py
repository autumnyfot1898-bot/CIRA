import copy
import json
from pathlib import Path
import tempfile
import unittest
import test_study_execution as fixture
from softcas.bindings import lock_development,bind_study,write_binding_outputs
from softcas.state import canonical
from softcas.study import GRID,SERIAL_RULES


class DevelopmentLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.cell='ControlledFog/YOLOv8n'
        a={'mask':[1,0,0,0,0,0],'ratio':.125,'lr':.001,'residual':.001}
        b={**a,'lr':.0003}
        self.spec={'protocol':'software_fixture','cells':{self.cell:{'local_candidates':{'A':a,'B':b}}},'records':[]}
        for family,candidates in [('local',['A','B']),('serial',list(SERIAL_RULES)),('controller',list(GRID)[:7])]:
            for index,candidate in enumerate(candidates):
                for seed in (9001,9002,9003):
                    score=.4+.01*index if family!='controller' else .5-(index-2)**2*.002
                    row={'role':'DEV','study_kind':'native_detector','complete':True,'seed':seed,'cell':self.cell,
                         'family':family,'candidate':candidate,'seen':score,'metric':'AP50:95','score_units':'fraction',
                         'selection_metric':'final_macro_seen','clients':4,'stages':4,'budget_identity':'shared-allowance',
                         'F_total':100.,'network_tx_bytes':0,'beta_lr':.01}
                    p=self.root/f'{family}_{candidate}_{seed}.json';p.write_bytes(canonical(row))
                    self.spec['records'].append(str(p))

    def test_dev_selection_uses_complete_seed_macro_seen_and_locks_q(self):
        lock=lock_development(self.spec)
        self.assertEqual(lock['selection']['controller_q'],'Q2')
        self.assertTrue(lock['selection']['locked'])
        self.assertEqual(lock['cells'][self.cell]['local_candidate'],'B')
        self.assertEqual(lock['cells'][self.cell]['fixed']['lr'],.0003)
        self.assertEqual(lock['selection']['serial_rules'][self.cell],SERIAL_RULES[-1])
        self.assertEqual(len(lock['inputs']),39)

    def test_missing_or_failed_candidate_cannot_be_selected_away(self):
        self.spec['records'].pop()
        with self.assertRaisesRegex(ValueError,'Missing or failed'):lock_development(self.spec)

    def test_T_data_rejected_and_budget_mismatch_not_hidden(self):
        p=Path(self.spec['records'][0]);row=json.loads(p.read_bytes());row['role']='T';p.write_bytes(canonical(row))
        with self.assertRaises(ValueError):lock_development(self.spec)
        row['role']='DEV';row['budget_identity']='more-work';p.write_bytes(canonical(row))
        with self.assertRaisesRegex(ValueError,'unequal budget'):lock_development(self.spec)

    def test_duplicate_seed_and_wrong_units_rejected(self):
        self.spec['records'].append(self.spec['records'][0])
        with self.assertRaisesRegex(ValueError,'Duplicate'):lock_development(self.spec)
        self.spec['records'].pop();p=Path(self.spec['records'][0]);row=json.loads(p.read_bytes());row['score_units']='percent';p.write_bytes(canonical(row))
        with self.assertRaises(ValueError):lock_development(self.spec)

    def test_native_study_registers_the_actual_locked_winner_and_pins_dev_records(self):
        from softcas.bindings import native_study
        lock=lock_development(self.spec)
        path=self.root/'lock.json';path.write_bytes(canonical(lock))
        spec={'datasets':['ControlledFog'],'detectors':['YOLOv8n'],'seeds':[1001],
              'development_lock':str(path)}
        study=native_study(spec)
        self.assertEqual(study['development_selection']['controller_q'],'Q2')
        self.assertEqual(study['native_development_lock']['path'],str(path))
        Path(self.spec['records'][0]).write_text('{}')
        with self.assertRaisesRegex(ValueError,'hash differs'):native_study(spec)

    def test_equal_scores_use_actual_computation_before_candidate_id(self):
        for filename in self.spec['records']:
            p=Path(filename);row=json.loads(p.read_bytes())
            if row['family']=='local':
                row['seen']=.5;row['F_total']=200 if row['candidate']=='A' else 100
            if row['family']=='controller':
                row['seen']=.5;row['F_total']=50 if row['candidate']=='Q4' else 100
            p.write_bytes(canonical(row))
        lock=lock_development(self.spec)
        self.assertEqual(lock['cells'][self.cell]['local_candidate'],'B')
        self.assertEqual(lock['selection']['controller_q'],'Q4')

    def test_serial_receiving_rate_must_match_the_selected_controller(self):
        for filename in self.spec['records']:
            p=Path(filename);row=json.loads(p.read_bytes())
            if row['family']=='serial':row['beta_lr']=.03;p.write_bytes(canonical(row))
        with self.assertRaisesRegex(ValueError,'receiving rate'):lock_development(self.spec)


class BindingInventoryTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture.StudyExecutionTests();self.f.setUp();self.addCleanup(self.f.doCleanups)

    def test_actual_inventory_materializes_a_bound_plan_without_execution(self):
        report=bind_study(self.f.study,self.f.spec)
        self.assertTrue(report['ready']);self.assertFalse(report['executed'])
        self.assertEqual(report['compiled_plan']['jobs'][0]['overrides']['ridge_lambda'],10.)
        files=write_binding_outputs(report,self.f.root/'binding')
        self.assertTrue(Path(files['plan']).is_file())
        self.assertEqual(json.loads(Path(files['plan']).read_bytes())['schema'],'cira.compiled_study.v1')
        with self.assertRaises(FileExistsError):write_binding_outputs(report,self.f.root/'binding')

    def test_missing_assets_return_exact_fields_and_no_ready_claim(self):
        self.f.spec['cells']['ControlledFog/YOLOv8n']['checkpoint']=None
        report=bind_study(self.f.study,self.f.spec)
        self.assertFalse(report['ready']);self.assertEqual(report['compiled_plan']['ready_trajectories'],0)
        self.assertTrue(any(r['field'].endswith('.checkpoint') for r in report['inventory_errors']))

    def test_compiled_job_metadata_tampering_is_rejected(self):
        from softcas.study_execution import preflight_job
        report=bind_study(self.f.study,self.f.spec)
        job=report['compiled_plan']['jobs'][0]
        job['interventions']={'donors':{'mode':'empty'}}
        with self.assertRaisesRegex(ValueError,'fields changed'):preflight_job(job)

    def test_extra_calibration_asset_is_pinned_at_execution(self):
        from softcas.study_execution import preflight_job
        path=self.f.root/'calibration.json';path.write_text('{}')
        self.f.spec['cells']['ControlledFog/YOLOv8n']['extra_inputs']=[str(path)]
        report=bind_study(self.f.study,self.f.spec)
        job=report['compiled_plan']['jobs'][0]
        path.write_text('changed')
        with self.assertRaisesRegex(ValueError,'changed before execution'):preflight_job(job)


class NativeMetadataTests(unittest.TestCase):
    def setUp(self):
        import hashlib
        from softcas.state import digest
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        labels=['person','rider','car','truck','bus','train','motorcycle','bicycle']
        self.records=[]
        for stage in range(1,5):
            for role in ('D','V','A'):
                key=f'{stage}-{role}'
                self.records.append(dict(image=key,image_sha256=digest(key),scene_id=key,role=role,domain=str(stage),client='0',stage=stage,verified_grouping=True))
        self.records.append(dict(image='test',image_sha256=digest('test'),scene_id='test',role='T',domain='1',client='0',stage=0,verified_grouping=True))
        self.base=self.root/'base.pt';self.base.write_bytes(b'test-metadata-only-not-weights')
        self.registry=self.root/'registry.json';self.registry.write_bytes(canonical({'labels':labels,'records':self.records}))
        self.detector=self.root/'detector.json';self.detector.write_bytes(canonical({'labels':labels,'ports':list(range(6)),'image_size':640}))
        self.profile=self.root/'cost.json'
        runtime={'family':'jetson_orin','precision':'float32','device':'cuda:0'}
        self.profile_data={'runtime':runtime,'runtime_hash':digest(runtime),'status':'validated','unknown_operators':{},
                           'base_sha256':hashlib.sha256(self.base.read_bytes()).hexdigest(),'models':{'test-only':{}}}
        self.profile.write_bytes(canonical(self.profile_data))
        self.config={'cost_profile':str(self.profile),'runtime':runtime}
        self.job={'checkpoint':str(self.base),'registry':str(self.registry),'detector_config':str(self.detector),'clients':['0']}

    def test_metadata_preflight_checks_declared_native_support_without_loading_weights(self):
        from softcas.study_execution import validate_native_metadata
        self.assertTrue(validate_native_metadata(self.job,self.config))

    def test_unvalidated_or_wrong_base_cost_profile_blocks_readiness(self):
        from softcas.study_execution import validate_native_metadata
        self.profile_data['status']='unvalidated';self.profile.write_bytes(canonical(self.profile_data))
        with self.assertRaisesRegex(ValueError,'validated'):validate_native_metadata(self.job,self.config)
        self.profile_data['status']='validated';self.profile_data['base_sha256']='a'*64;self.profile.write_bytes(canonical(self.profile_data))
        with self.assertRaisesRegex(ValueError,'different frozen base'):validate_native_metadata(self.job,self.config)

    def test_missing_stage_role_and_cross_role_scene_leakage_block_readiness(self):
        from softcas.study_execution import validate_native_metadata
        self.records[0]['role']='DEV';self.registry.write_bytes(canonical({'labels':json.loads(self.detector.read_bytes())['labels'],'records':self.records}))
        with self.assertRaisesRegex(ValueError,'support missing'):validate_native_metadata(self.job,self.config)
