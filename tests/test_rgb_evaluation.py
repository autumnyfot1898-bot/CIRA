import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch

from softcas.data import ManifestSplit, detection_record_batch
from softcas.rgb_evaluation import perturbed_split


class RGBEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        file = self.root / "input.png"
        Image.fromarray(np.arange(7*5*3,dtype=np.uint8).reshape(7,5,3)).save(file)
        self.record = dict(image=file.name,image_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                           boxes=[[0,.5,.5,.4,.4]],domain="fog",role="T",client="0",stage=1)
        self.source = ManifestSplit("T","0",1,"fog",[self.record],self.root,["car"],16,1)

    def test_zero_noise_is_exact_clean_preprocessing(self):
        clean = self.source.batches[0]
        noised = perturbed_split(self.source,0.,1001).batches[0]
        self.assertTrue(torch.equal(clean["img"],noised["img"]))
        self.assertTrue(torch.equal(clean["bboxes"],noised["bboxes"]))
        self.assertEqual(clean["img"].dtype,torch.float32)

    def test_shared_image_noise_is_replayable_and_changes_pixels_not_boxes(self):
        first = perturbed_split(self.source,.03,1001).batches[0]
        again = perturbed_split(self.source,.03,1001).batches[0]
        other = perturbed_split(self.source,.03,1002).batches[0]
        self.assertTrue(torch.equal(first["img"],again["img"]))
        self.assertFalse(torch.equal(first["img"],other["img"]))
        self.assertTrue(torch.equal(first["bboxes"],self.source.batches[0]["bboxes"]))
        self.assertTrue(torch.all((first["img"]>=0)&(first["img"]<=1)))

    def test_noise_identity_prevents_clean_cache_reuse(self):
        identities={perturbed_split(self.source,s,1001).identity for s in (0.,.01,.03)}
        self.assertEqual(len(identities),3)
        self.assertNotIn(self.source.identity,identities)

    def test_noise_key_omits_method_profile_and_path(self):
        record = dict(self.record)
        copied = self.root / "renamed.png"
        copied.write_bytes((self.root/self.record["image"]).read_bytes())
        record["image"]=str(copied)
        left=detection_record_batch(self.record,self.root,16,noise_sigma=.01,noise_seed=1001)
        right=detection_record_batch(record,self.root,16,noise_sigma=.01,noise_seed=1001)
        self.assertTrue(torch.equal(left["img"],right["img"]))

    def test_invalid_noise_and_non_test_roles_rejected(self):
        for sigma in (True,float("nan"),-.1):
            with self.assertRaises(ValueError): detection_record_batch(self.record,self.root,16,noise_sigma=sigma)
        self.source.role="D"
        with self.assertRaises(ValueError): perturbed_split(self.source,.01,1001)


if __name__=="__main__":unittest.main()
