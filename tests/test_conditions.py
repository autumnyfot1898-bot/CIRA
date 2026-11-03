import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from softcas.conditions import (build_condition, contact_condition, donor_condition, domain_order,
                                nested_d, scene_alpha)
from softcas.data import validate_registry
from softcas.state import canonical, digest


def registry():
    rows = []
    for role in ("D", "V", "A", "T", "DEV"):
        for domain in ("fog", "rain"):
            for i in range(32 if role in ("D", "V", "A") else 2):
                identity = f"{role}-{domain}-{i}"
                rows.append(dict(image=str(Path(tempfile.gettempdir()) / (identity + ".png")),
                    image_sha256=hashlib.sha256(identity.encode()).hexdigest(), scene_id=identity,
                    role=role, domain=domain, client=str(i % 2) if role in ("D", "V", "A") else "shared",
                    stage=(1 if domain == "fog" else 2), verified_grouping=True,
                    boxes=[[i % 8, .5, .5, .2, .2]]))
    return {"schema": 1, "labels": ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"], "records": rows}


def trace():
    return [dict(receiver="0", owner="1", offset_seconds=t, online=online)
            for t, online in [(0, True), (1, True), (3, False), (4, True), (7, False), (8, True), (9, False), (10, True)]]


class ConditionTests(unittest.TestCase):
    def test_nested_scene_sets_and_heldout_unchanged(self):
        source = registry()
        results = [nested_d(source, f, 1001)[0] for f in (.5, .75, 1.)]
        sets = [{r["scene_id"] for r in v["records"] if r["role"] == "D"} for v in results]
        self.assertLess(sets[0], sets[1])
        self.assertLess(sets[1], sets[2])
        heldout = [r for r in source["records"] if r["role"] != "D"]
        for result in results:
            self.assertEqual([r for r in result["records"] if r["role"] != "D"], heldout)
        self.assertEqual(results[2], source)

    def test_whole_scene_is_not_partly_removed(self):
        source = registry()
        extra = copy.deepcopy(source["records"][0])
        extra.update(image=extra["image"]+"-second", image_sha256="e"*64)
        source["records"].append(extra)
        result, evidence = nested_d(source, .5, 1001)
        count = sum(r["scene_id"] == extra["scene_id"] for r in result["records"])
        self.assertIn(count, (0, 2))
        self.assertTrue(evidence["cells"])

    def test_order_changes_stages_only(self):
        source = registry()
        result, evidence = domain_order(source, ["fog", "rain"], ["0", "1"], 1001, "common", ["rain", "fog"])
        self.assertEqual(evidence["orders"]["0"], evidence["orders"]["1"])
        for before, after in zip(source["records"], result["records"]):
            self.assertEqual({k:v for k,v in before.items() if k != "stage"}, {k:v for k,v in after.items() if k != "stage"})
            if before["role"] in ("T", "DEV"):
                self.assertEqual(before, after)
            else:
                self.assertEqual(after["stage"], 1 if after["domain"] == "rain" else 2)
        validate_registry(result["records"], formal=True)

    def test_alpha_uses_shared_targets_and_real_assignment(self):
        source = registry()
        result, evidence = scene_alpha(source, ["fog", "rain"], ["0", "1"], 1001, "equal", 2)
        self.assertEqual(evidence["targets"], [[.5, .5]] * 8)
        self.assertEqual(len(evidence["diagnostics"]), 6)
        self.assertTrue(all(d["targets"] == evidence["targets"] for d in evidence["diagnostics"]))
        self.assertEqual([r for r in source["records"] if r["role"] in ("T", "DEV")],
                         [r for r in result["records"] if r["role"] in ("T", "DEV")])
        self.assertNotEqual([r["client"] for r in source["records"]], [r["client"] for r in result["records"]])

    def test_half_visibility_masks_opportunities_not_individual_true_events(self):
        events = trace()
        result, evidence, intervention = contact_condition(events, 1001, "half_visibility")
        self.assertEqual(evidence["eligible_opportunity_count"], 4)
        self.assertEqual(evidence["retained_opportunity_count"], 2)
        self.assertEqual(result[0]["online"], result[1]["online"])
        self.assertEqual(intervention, {})
        self.assertTrue(all(not after["online"] or before["online"] for before, after in zip(events, result)))
        self.assertEqual(result, contact_condition(events, 1001, "half_visibility")[0])

    def test_half_payload_is_one_explicit_contact_not_probability(self):
        event = dict(stage=2, receiver="0", owner="1", contact_offset_seconds=4)
        result, evidence, intervention = contact_condition(trace(), 1001, "one_stage2_half_payload", event)
        self.assertEqual(result, trace())
        self.assertEqual(intervention["payload_fraction"], .5)
        self.assertEqual(intervention["kind"], "single_half_payload")
        self.assertNotIn("probability", intervention)
        self.assertEqual(intervention, evidence["interruption"])
        event["contact_offset_seconds"] = 5
        with self.assertRaises(ValueError): contact_condition(trace(), 1001, "one_stage2_half_payload", event)

    def test_weak_donor_requires_dev_exact_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"weak.json"
            lock = dict(schema="cira.weak_donor_lock.v1", role="T", allowlist=["a"*64], construction="low performance")
            path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaises(ValueError): donor_condition("weak", str(path))
            lock.update(role="DEV", construction="DEV fixed lower-data donor checkpoint set")
            path.write_text(json.dumps(lock), encoding="utf-8")
            policy, evidence = donor_condition("weak", str(path))
            self.assertEqual(policy["mode"], "allowlist")
            self.assertEqual(policy["allowlist"], ["a"*64])
            self.assertEqual(evidence["development_lock"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_materialization_has_hash_bound_outputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, contacts = root/"source.json", root/"trace.json"
            source.write_bytes(canonical(registry()));contacts.write_bytes(canonical(trace()))
            initial_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            condition=dict(dataset="test", detector="test", seed=1001, alpha="equal", data_fraction=.5,
                           order="common", contacts="half_visibility", donors="empty")
            spec=dict(condition=condition, source_registry=str(source), source_contact_trace=str(contacts),
                      domains=["fog", "rain"], clients=["0", "1"], max_passes=2, formal=True)
            result = build_condition(spec, root/"out")
            self.assertEqual(result["condition_key"], digest(condition))
            self.assertTrue(result["complete"])
            self.assertEqual(result["interventions"]["donors"], {"mode":"empty"})
            self.assertEqual(result["outputs"]["registry"]["sha256"], hashlib.sha256((root/"out/registry.json").read_bytes()).hexdigest())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), initial_hash)
            self.assertTrue(all(r["images"] > 0 for r in result["evidence"]["achieved_histograms"]))
            with self.assertRaises(FileExistsError): build_condition(spec, root/"out")

    def test_bad_sources_and_malformed_conditions_rejected(self):
        with self.assertRaises(ValueError): nested_d(registry(), 0, 1001)
        with self.assertRaises(ValueError): nested_d(registry(), True, 1001)
        with self.assertRaises(ValueError): domain_order(registry(), ["fog","rain"], ["0","1"], 1, "common", ["fog"])
        with self.assertRaises(ValueError): contact_condition([trace()[0], trace()[0]], 1, "normal")
        with self.assertRaises(ValueError): scene_alpha(registry(), ["fog","rain"], ["0","1"], 1, 0, 1)


if __name__ == "__main__":
    unittest.main()
