import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import verify


class VerifierProtocolTests(unittest.TestCase):
    def test_default_checks_do_not_load_historical_interfaces_or_suites(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch('verify.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='',stderr='Ran 3 tests\nOK')), \
             patch('verify.check_suite') as suite, patch('verify.provenance',return_value={}):
            verify.main(['--output',folder])
            suite.assert_not_called()
            value=json.loads(Path(folder,'verification.json').read_bytes())
            self.assertEqual(value['unit_tests']['count'],3)
            self.assertFalse(value['historical_interface_reports_used'])
            self.assertIsNone(value['suite'])

    def test_existing_evidence_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            old=Path(folder,'verification.json');old.write_bytes(b'previous evidence')
            with patch('verify.subprocess.run') as run, self.assertRaises(FileExistsError):
                verify.main(['--output',folder])
            run.assert_not_called()
            self.assertEqual(old.read_bytes(),b'previous evidence')

    def test_only_explicit_fixture_suite_is_replayed(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch('verify.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='',stderr='Ran 3 tests\nOK')), \
             patch('verify.check_suite',return_value={'scope':'fixture'}) as suite, \
             patch('verify.provenance',return_value={}):
            verify.main(['--output',folder,'--suite','specific-fixture'])
            suite.assert_called_once_with(Path('specific-fixture'))
