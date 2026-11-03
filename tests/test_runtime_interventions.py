import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from test_invariants import Fixture
from softcas.cache import PackageCache
from softcas.interventions import RuntimeInterventions, validate_interventions
from softcas.resources import Cost, ExecutionStopped, Ledger
from softcas.state import canonical, digest
from softcas.transport import PeerService, PeerServer, LiveExchange


class RuntimeInterventionTests(Fixture):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.trace = [dict(receiver='0', owner='peer', offset_seconds=0., online=True)]
        self.origin = time.time()-1

    def policy(self, mode='normal', half=False):
        donor = {} if mode == 'normal' else {'mode': mode}
        if mode == 'allowlist':
            lock = {'schema': 'cira.weak_donor_lock.v1', 'role': 'DEV',
                    'allowlist': [self.package().hash], 'construction': 'fixed DEV versions'}
            path = self.root/'weak.json';path.write_bytes(canonical(lock))
            donor.update(allowlist=lock['allowlist'], role='DEV',
                         development_lock={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        event = {'kind':'single_half_payload','stage':2,'receiver':'0','owner':'peer',
                 'contact_offset_seconds':0.,'payload_fraction':.5,'event_id':'a'*64} if half else {}
        return {'donors': donor, 'contacts': event}

    def exchange_pair(self, policy):
        runtime=RuntimeInterventions(policy,self.root/'event.json',self.trace,self.origin)
        cache=PackageCache(1000000,self.backend.specs,allowed_package=runtime.allows)
        service=PeerService('peer',self.source,'CIRA_TEST_TOKEN',self.root/'published',1000000,interventions=runtime)
        package=self.package();service.publish(package)
        server=PeerServer(('127.0.0.1',0),service,3)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        exchange=LiveExchange('0',cache,self.ledger,{'peer':{'host':'127.0.0.1','port':server.server_address[1]}},
                              'CIRA_TEST_TOKEN',timeout=3,interventions=runtime,stage=2)
        return runtime,cache,service,exchange,package

    def test_empty_donor_blocks_catalogue_cache_and_inherited_state(self):
        runtime,cache,service,exchange,p=self.exchange_pair(self.policy('empty'))
        with patch.dict(os.environ,{'CIRA_TEST_TOKEN':'test-only'}), patch.object(exchange,'_client') as client:
            exchange.refresh();client.assert_not_called()
        self.assertFalse(exchange.catalogue);self.assertFalse(cache.keys())
        with self.assertRaises(ExecutionStopped):cache.put(p.hash,p.wire)
        with self.assertRaises(ExecutionStopped):exchange.acquire(p.hash,time.time())
        with self.assertRaises(ValueError):runtime.require_state(replace(self.state,C=(p.hash,)))
        self.assertEqual(self.ledger.used['search'].B,0)

    def test_weak_donor_real_catalogue_filters_before_limit_and_acquires(self):
        runtime,cache,service,exchange,p=self.exchange_pair(self.policy('allowlist'))
        other=self.package(seed=10);service.publish(other)
        with patch.dict(os.environ,{'CIRA_TEST_TOKEN':'test-only'}):
            exchange.refresh()
            self.assertEqual(set(exchange.catalogue),{p.hash})
            self.assertEqual(exchange.acquire(p.hash,time.time()).hash,p.hash)
            with self.assertRaises(ExecutionStopped):exchange.acquire(other.hash,time.time())
        self.assertEqual(cache.keys(),(p.hash,))
        self.assertEqual([x['hash'] for x in service.catalogue(1,[p.hash])['versions']],[p.hash])

    def test_half_payload_real_socket_once_and_retry_from_zero(self):
        runtime,cache,service,exchange,p=self.exchange_pair(self.policy(half=True))
        with patch.dict(os.environ,{'CIRA_TEST_TOKEN':'test-only'}):
            exchange.refresh()
            with self.assertRaises(ConnectionError):exchange.acquire(p.hash,time.time())
            self.assertNotIn(p.hash,cache)
            claims=runtime.summary()['events']
            self.assertEqual(claims['a'*64]['actual_payload_bytes'],len(p.wire)//2)
            self.assertEqual(claims['a'*64]['status'],'interrupted_at_half')
            self.assertEqual(exchange.acquire(p.hash,time.time()).hash,p.hash)
        payloads=[x for x in service.records if x.get('leg','').endswith(':payload:tx')]
        self.assertEqual(sorted(x['bytes'] for x in payloads),[len(p.wire)//2,len(p.wire)])
        receive=[x for x in exchange.records if x.get('leg','').endswith(':payload:rx')]
        self.assertEqual(sum(x['bytes'] for x in receive),sum(x['bytes'] for x in payloads))
        self.assertEqual(len([x for x in service.records if x.get('kind')=='registered_half_payload']),1)
        self.assertEqual(service.remaining_quota,1000000-len(p.wire)//2-len(p.wire))

    def test_claim_survives_reload_and_different_condition_rejected(self):
        spec=self.policy(half=True);path=self.root/'durable.json'
        runtime=RuntimeInterventions(spec,path,self.trace,self.origin)
        self.assertEqual(runtime.payload_limit('0','peer',2,101,'request'),50)
        restored=RuntimeInterventions(spec,path,self.trace,self.origin)
        self.assertIsNone(restored.payload_limit('0','peer',2,101,'retry'))
        with self.assertRaises(ValueError):RuntimeInterventions(self.policy('empty'),path,self.trace,self.origin)

    def test_wrong_stage_or_contact_does_not_consume_event(self):
        runtime=RuntimeInterventions(self.policy(half=True),trace=self.trace,origin=self.origin)
        self.assertIsNone(runtime.payload_limit('0','peer',1,100,'stage1'))
        self.assertIsNone(runtime.payload_limit('1','peer',2,100,'other'))
        self.assertIsNone(runtime.payload_limit('0','peer',2,100,'past',now=self.origin-1))
        self.assertFalse(runtime.summary()['events'])
        self.assertEqual(runtime.payload_limit('0','peer',2,100,'right'),50)

    def test_changed_dev_lock_cannot_authorize_donors(self):
        spec=self.policy('allowlist')
        Path(spec['donors']['development_lock']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'lock changed'):validate_interventions(spec)

    def test_normal_policy_preserves_default_tcp_behavior(self):
        runtime,cache,service,exchange,p=self.exchange_pair({})
        with patch.dict(os.environ,{'CIRA_TEST_TOKEN':'test-only'}):
            exchange.refresh();self.assertEqual(exchange.acquire(p.hash,time.time()).hash,p.hash)
        self.assertFalse(runtime.summary()['events'])
