import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from softcas.external import validate_adapter, execute_external, source_identity
from softcas.state import canonical, digest
from softcas.interventions import validate_interventions


RUNNER = '''import argparse,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--job');p.add_argument('--result');p.add_argument('--mode',default='ok');a=p.parse_args()
j=json.loads(Path(a.job).read_bytes())
if a.mode=='timeout':time.sleep(30)
if a.mode=='exit':raise SystemExit(2)
if a.mode=='invalid':Path(a.result).write_text('{}');raise SystemExit(0)
matrix=[dict(seed=j['seed'],client='0',stage=1,current_domain='clear',domain='clear',deployed='test-state',score=.25)]
matrix_path=Path(a.result).with_name('matrix.json');matrix_path.write_text(json.dumps(matrix))
r=dict(schema='cira.external_trajectory.v1',method=j['method'],seed=j['seed'],cell=j['dataset']+'/'+j['detector'],
 complete=True,metric='test_quality_NOT_AP',score_units='fraction',input_manifest=j.get('input_manifest',{}),
 interventions_sha256=hashlib.sha256(json.dumps({'donors':{},'contacts':{}},sort_keys=True,separators=(',',':')).encode()).hexdigest(),
 matrix={'path':str(matrix_path),'sha256':hashlib.sha256(matrix_path.read_bytes()).hexdigest()},
 F_total=1.,network_tx_bytes=0,selection_E=1,max_client_update_seconds=.1)
for k in ['study_trajectory_id','study_recipe']:
 if k in j:r[k]=j[k]
if a.mode=='wrong_method':r['method']='softcas'
if a.mode=='wrong_units':r['score_units']='percent'
if a.mode=='summary':r['seen']=.9
if a.mode=='wrong_recipe':r['study_trajectory_id']='wrong'
Path(a.result).write_text(json.dumps(r))
'''


class ExternalExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'source';self.source.mkdir()
        self.script=self.source/'runner.py';self.script.write_text(RUNNER,encoding='utf8')
        self.adapter={'schema':'cira.external_adapter.v1','method':'DPFL',
                      'source':{'root':str(self.source),'repository_url':'https://example.org/test-only',
                                'commit':'a'*40,'files':{'runner.py':self.sha(self.script)}},
                      'runner':{'argv':[sys.executable,str(self.script),'--job','{job}','--result','{result}'],
                                'cwd':str(self.source),'files':[self.binding(self.script)],'timeout_seconds':10}}
        self.qualification=self.root/'qualification.json'
        self.qualification.write_bytes(canonical({'schema':'cira.external_qualification.v1','method':'DPFL',
                                       'source_identity':source_identity(self.adapter['source']),'validated':False}))
        self.adapter['qualification']=self.binding(self.qualification)
        self.job={'method':'DPFL','seed':1001,'dataset':'ControlledFog','detector':'YOLOv8n','clients':['0'],
                  'formal':False,'output':str(self.root/'out'),'adapter':self.adapter,'input_manifest':{}}

    @staticmethod
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

    def binding(self,path):return {'path':str(path),'sha256':self.sha(path)}

    def mode(self,value):self.adapter['runner']['argv']+=['--mode',value]

    def test_actual_subprocess_and_matrix_summary_and_completed_resume(self):
        validation=validate_adapter(self.adapter,self.job)
        self.assertEqual(validation['method'],'DPFL')
        result=execute_external(self.job)
        self.assertTrue(result['complete']);self.assertEqual(result['seen'],.25)
        self.assertEqual(result,execute_external(self.job,resume=True))
        with self.assertRaises(FileExistsError):execute_external(self.job)

    def test_no_internal_alias_and_no_shell_runner(self):
        self.adapter['runner']['argv']=[sys.executable,'-m','softcas.production','{job}','{result}']
        with self.assertRaisesRegex(ValueError,'alias'):validate_adapter(self.adapter,self.job)

    def test_source_tampering_and_unbound_imported_file_rejected(self):
        self.script.write_text('changed')
        with self.assertRaisesRegex(ValueError,'content changed'):validate_adapter(self.adapter,self.job)
        self.script.write_text(RUNNER,encoding='utf8')
        (self.source/'other.py').write_text('pass')
        with self.assertRaisesRegex(ValueError,'Unbound executable'):validate_adapter(self.adapter,self.job)

    def test_unqualified_fixture_cannot_become_formal(self):
        self.job['formal']=True
        with self.assertRaises(ValueError):validate_adapter(self.adapter,self.job)

    def test_runner_failure_keeps_missing_scores_and_partial_execution_record(self):
        self.mode('exit');result=execute_external(self.job)
        self.assertFalse(result['complete']);self.assertIsNone(result['seen']);self.assertFalse(result['cost_complete'])
        with self.assertRaisesRegex(RuntimeError,'Reconcile'):execute_external(self.job,resume=True)

    def test_timeout_is_recorded_not_silently_retried(self):
        self.mode('timeout');self.adapter['runner']['timeout_seconds']=.15
        result=execute_external(self.job)
        self.assertFalse(result['complete']);self.assertEqual(result['failure'],'timeout')

    def test_wrong_method_units_and_summary_are_not_accepted(self):
        for mode in ('wrong_method','wrong_units','summary','invalid'):
            with self.subTest(mode=mode):
                self.job['output']=str(self.root/mode)
                self.adapter['runner']['argv']=self.adapter['runner']['argv'][:6]
                self.mode(mode)
                result=execute_external(self.job)
                self.assertFalse(result['complete']);self.assertIsNone(result['seen'])

    def test_changed_resume_inputs_are_not_reused(self):
        execute_external(self.job)
        self.adapter['runner']['timeout_seconds']=20
        with self.assertRaisesRegex(ValueError,'identity changed'):execute_external(self.job,resume=True)

    def test_source_path_escape_is_rejected(self):
        self.adapter['source']['files']={'../qualification.json':self.sha(self.qualification)}
        with self.assertRaisesRegex(ValueError,'escapes'):validate_adapter(self.adapter,self.job)

    def test_bound_study_dispatches_real_external_process(self):
        import test_study_execution as sample
        from softcas.study_execution import compile_study,execute_study,preflight_job
        f=sample.StudyExecutionTests();f.setUp();self.addCleanup(f.doCleanups)
        f.spec['selection']={'blocks':['N01'],'methods':['DPFL']}
        f.spec['clients']=['0'];f.spec['external_adapters']={'DPFL':self.adapter}
        plan=compile_study(f.study,f.spec)
        self.assertFalse(plan['blocked']);self.assertEqual(plan['jobs'][0]['executor'],'external')
        self.assertIn(str(self.script),plan['jobs'][0]['input_manifest'])
        result=execute_study(plan)
        self.assertTrue(result[0]['complete']);self.assertEqual(result[0]['seen'],.25)
        self.assertEqual(result[0]['study_trajectory_id'],plan['jobs'][0]['study_trajectory_id'])

    def test_external_physical_plan_uses_one_explicit_coordinator(self):
        import test_study_execution as sample
        from softcas.study_execution import compile_study
        f=sample.StudyExecutionTests();f.setUp();self.addCleanup(f.doCleanups)
        f.spec.update(selection={'blocks':['N01'],'methods':['DPFL']},clients=['0','1','2','3'],
                      external_adapters={'DPFL':self.adapter},external_coordinator_id='2',epoch_start_unix=1900000000,
                      physical_nodes={str(i):{'endpoint':{'host':'127.0.0.1','port':18000+i}} for i in range(4)})
        plan=compile_study(f.study,f.spec)
        self.assertFalse(plan['blocked']);self.assertEqual(len(plan['jobs']),1)
        job=plan['jobs'][0]
        self.assertEqual(job['node_id'],'2');self.assertEqual(job['clients'],['0','1','2','3'])
        self.assertEqual(job['external_topology'],'runner_managed_complete_federation')
        self.assertFalse(plan['trajectory_groups'])

    def test_process_fixture_is_not_counted_as_a_native_study_endpoint(self):
        import test_study_execution as sample
        from softcas.study_execution import compile_study,execute_study
        from softcas.study import summarize_completed
        f=sample.StudyExecutionTests();f.setUp();self.addCleanup(f.doCleanups)
        f.spec.update(selection={'blocks':['N01'],'methods':['DPFL']},clients=['0'],external_adapters={'DPFL':self.adapter})
        result=execute_study(compile_study(f.study,f.spec))
        summary=summarize_completed(f.study,result)
        self.assertTrue(all(r['complete']==0 for r in summary['blocks']))

