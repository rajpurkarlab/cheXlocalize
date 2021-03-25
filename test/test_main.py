import os
import shutil
import unittest
from glob import glob
from os.path import join

from main import train, test


class TestTrainFunc(unittest.TestCase):
    path = os.path.abspath("./sandbox")

    def setUp(self):
        self.train = lambda x: train(exp_name=x,
                                     save_dir=self.path,
                                     gpus=None,
                                      weights_summary=None)

    def test_default(self):
        self.train("TEST_DEFAULT")

    def test_slurm(self):
        os.environ['SLURM_JOB_ID'] = "138"
        self.train("TEST_SLURM")
        self.assertTrue(os.path.exists(join(self.path,
                        "TEST_SLURM/lightning_logs/version_0")))

    def test_override(self):
        self.train("TEST_OVERRIDE")
        with self.assertRaises(FileExistsError):
            self.train("TEST_OVERRIDE")
    
class TestTestFunc(unittest.TestCase):
    path = os.path.abspath("./sandbox")

    def setUp(self):
        self.train = lambda x: train(exp_name=x,
                                   save_dir=self.path,
                                   gpus=None,
                                   weights_summary=None)
        self.test = lambda x: test(ckpt_path=join(self.path,
                                                  f"{x}/ckpts.ckpt"),
                                   gpus=None)

    def test_default(self):
        self.train("TEST_TESTING_DEFAULT")
        self.test("TEST_TESTING_DEFAULT")
    
if __name__ == "__main__":
    unittest.main(verbosity=0)
    shutil.rmtree(os.path.abspath("./sandbox"), ignore_errors=True)
