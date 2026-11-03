"""Regression checks for exact checkpoint recovery and atomic replacement."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from softcas.state import FitSettings, State, Structure
from softcas.storage import BoundedStore, CheckpointStore


def checkpoint(label):
    return State(Structure((1, 0, 0, 0, 0, 0), 1 / 8),
                 FitSettings(1e-3, 1e-3), (), (), (), '{}').lineage(label=label)


class CheckpointIdentityRepairs(unittest.TestCase):
    def test_index_cannot_substitute_another_valid_checkpoint(self):
        a, b = checkpoint('A'), checkpoint('B')
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = CheckpointStore(BoundedStore(directory, 100000))
            checkpoints.save(a)
            b_key = checkpoints.save(b)
            # Both content-addressed objects and their embedded hashes are valid.
            # The index must not be trusted to bind a request to the right one.
            checkpoints.index[a.hash] = b_key
            with self.assertRaisesRegex(ValueError, 'identity|hash'):
                checkpoints.load(a.hash)

    def test_file_change_after_verified_read_does_not_substitute_state(self):
        state, other = checkpoint('complete evaluated state'), checkpoint('another state')
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = CheckpointStore(BoundedStore(directory, 100000))
            key = checkpoints.save(state)
            other_key = checkpoints.save(other)
            other_payload = checkpoints.store.get(other_key)
            get = checkpoints.store.get

            def read_then_change_file(object_key):
                verified = get(object_key)
                # Simulate the file changing after validation but before decode.
                checkpoints.store._path(object_key).write_bytes(other_payload)
                return verified

            with patch.object(checkpoints.store, 'get', side_effect=read_then_change_file):
                self.assertEqual(checkpoints.load(state.hash), state)

    def test_store_roundtrip_after_restart(self):
        state = checkpoint('restart')
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = CheckpointStore(BoundedStore(directory, 100000))
            checkpoints.save(state)
            restarted = CheckpointStore(BoundedStore(directory, 100000))
            self.assertEqual(restarted.load(state.hash), state)

    def test_failed_atomic_replacement_preserves_previous_checkpoint(self):
        a, b = checkpoint('previous'), checkpoint('replacement')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'deployed.json'
            a.save(path)
            original = path.read_bytes()
            with patch('softcas.storage.os.replace', side_effect=OSError('disk replacement failed')):
                with self.assertRaises(OSError):
                    b.save(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(State.load(path), a)
            self.assertEqual(list(Path(directory).glob('.pending-*')), [])

    def test_successful_atomic_replacement_writes_complete_state(self):
        a, b = checkpoint('previous'), checkpoint('replacement')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'deployed.json'
            a.save(path)
            b.save(path)
            self.assertEqual(State.load(path), b)


if __name__ == '__main__':
    unittest.main()
