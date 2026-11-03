"""Learning checks on real CPU tensor execution, not synthetic gain labels."""
import unittest
import torch

from softcas.cache import PackageCache
from softcas.engine import initial_state
from softcas.state import Structure, FitSettings
from softcas.toy_learning import TeacherBackend, teacher_split, fitting_probe


class LearningSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setup_fixture(self):
        backend = TeacherBackend()
        cache = PackageCache(1000000, backend.specs)
        state = initial_state(backend, cache, Structure((1, 0, 0, 0, 0, 0), 1/8),
                              FitSettings(1e-3, 1e-4), 42)
        return backend, cache, state

    def test_real_positive_and_negative_generalization_probe(self):
        backend, cache, state = self.setup_fixture()
        V = teacher_split("V", "probe", 1, "0", 500)
        results = []
        for direction in (1.0, -1.0):
            child, probe = fitting_probe(backend, state, cache,
                teacher_split("D", "probe", 1, "0", 200, target_direction=direction), V,
                steps=32, optimizer_settings={"name": "AdamW", "weight_decay": 0.0})
            self.assertTrue(probe["base_unchanged"])
            self.assertGreater(probe["local_parameter_delta_l2"], 0)
            self.assertGreater(probe["steps"][0]["gradient_l2"], 0)
            self.assertLess(probe["D_loss_after"], probe["D_loss_before"])
            self.assertNotEqual(child.phi, state.phi)
            self.assertEqual(child.z, state.z)
            results.append(probe)
        self.assertGreater(results[0]["V_gain"], 0)
        self.assertLess(results[1]["V_gain"], 0)

    def test_role_identity_and_repeated_score(self):
        backend, cache, state = self.setup_fixture()
        D = teacher_split("D", "0", 1, "0", 100)
        V = teacher_split("V", "0", 1, "0", 101)
        self.assertNotEqual(D.identity, V.identity)
        self.assertFalse(torch.equal(D.batches[0]["img"], V.batches[0]["img"]))
        model = backend.assembly(state, cache, mode="import")
        self.assertEqual(backend.score(model, V), backend.score(model, V))
        with torch.no_grad():
            errors = [(model(b["img"]).double() - b["target"].double()).flatten() for b in V.batches]
            expected = 1 / (1 + float(torch.cat(errors).square().mean()))
        self.assertAlmostEqual(backend.score(model, V), expected, places=14)

    def test_generator_does_not_change_global_random_state(self):
        before = torch.random.get_rng_state().clone()
        first = teacher_split("D", "0", 1, "0", 100)
        second = teacher_split("D", "0", 1, "0", 100)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(first.identity, second.identity)

    def test_invalid_role_and_direction_rejected(self):
        with self.assertRaises(ValueError):
            teacher_split("REPORT", "0", 1, "0", 100)
        with self.assertRaises(ValueError):
            teacher_split("D", "0", 1, "0", 100, target_direction=0)


if __name__ == "__main__":
    unittest.main()
