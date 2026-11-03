import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from softcas.measurement import summarize_sessions
from softcas.device import RemoteReleaseService
from softcas.resources import Ledger, Cost
from softcas.state import canonical


class DeviceSessionProtocolTests(unittest.TestCase):
    def report(self, job, p95):
        return dict(candidate='same-checkpoint', job_id=job, device=dict(
            p95_ms=p95, p50_ms=p95/2, maximum_ms=p95+1,
            peak_process_ram_bytes=100+p95, repeats=200, warmup=50,
            physical_device_id='fixture_device', executor='torch', latency_samples_ms=[p95]))

    def test_within_seed_summary_preserves_repetition_boundary(self):
        values=[self.report(str(i), p) for i,p in enumerate([10,20,30])]
        result=summarize_sessions(values)
        self.assertEqual(result['p95_ms'],20)
        self.assertEqual(result['maximum_ms'],31)
        self.assertEqual(result['peak_process_ram_bytes'],130)
        self.assertEqual(result['timed_requests'],600)
        self.assertEqual(result['warmup_requests'],150)
        self.assertEqual(result['measurement_sessions'],3)
        self.assertNotIn('latency_samples_ms',result)
        self.assertIn('latency_samples_ms',values[0]['device'])

    def test_duplicate_job_and_mixed_checkpoint_are_rejected(self):
        a=self.report('first',10)
        with self.assertRaises(ValueError):summarize_sessions([a,a])
        b=self.report('second',20);b['candidate']='other'
        with self.assertRaises(ValueError):summarize_sessions([a,b])
        b=self.report('second',20);b['device']['physical_device_id']='other'
        with self.assertRaises(ValueError):summarize_sessions([a,b])

    def test_invalid_latency_is_not_averaged(self):
        for v in [float('nan'),float('inf'),-1]:
            with self.assertRaises(ValueError):summarize_sessions([self.report('first',v)])

    def test_session_identity_is_distinct_and_retry_is_idempotent(self):
        service=RemoteReleaseService({},'registry','0')
        ledger=Ledger(Cost(F=1e9),Cost())
        state=SimpleNamespace(hash='checkpoint')
        split=SimpleNamespace(client='0',stage=1,domain='fixture',role='T',identity='data')
        ids=[]
        class Rpc:
            def call(self, operation, request, *_):
                ids.append(request['request_id'])
                return dict(candidate=state.hash,data_identity=split.identity,job_id=request['request_id'],
                    actual_cost=dict(F=0),counting_coverage_complete=True,candidate_score=None),b''
        with patch('softcas.device.bundle_bytes',side_effect=lambda *args:canonical(args[-1])), \
             patch.object(service,'_rpc',return_value=Rpc()):
            for repeat in [0,1,0]:
                service.evaluate(None,state,split,None,ledger,{},measure=True,measurement_repeat=repeat)
            with self.assertRaises(ValueError):
                service.evaluate(None,state,split,None,ledger,{},measure=False,measurement_repeat=0)
        self.assertNotEqual(ids[0],ids[1])
        self.assertEqual(ids[0],ids[2])
