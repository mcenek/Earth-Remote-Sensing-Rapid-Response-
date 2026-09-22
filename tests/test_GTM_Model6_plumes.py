import sys, unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/GTM_Model6'))
from GTM_Model6_plumes import PlumeUNet, dense_loss, features, predict

class PlumeContractTests(unittest.TestCase):
    def test_unknown_emit_exterior_cannot_change_loss(self):
        p=torch.zeros((1,1,16,16));y=torch.zeros_like(p);known=torch.zeros_like(p);known[:,:,4:8,4:8]=1;y[:,:,4:8,4:8]=1
        changed=y.clone();changed[known==0]=1
        self.assertEqual(float(dense_loss(p,y,known,torch.zeros(1))),float(dense_loss(p,changed,known,torch.zeros(1))))
    def test_input_feature_radiometric_scale_invariance(self):
        rng=np.random.default_rng(6);a=rng.integers(500,2000,(5,32,32)).astype('float32');b=rng.integers(500,2000,(5,32,32)).astype('float32');v=np.ones((32,32),bool)
        np.testing.assert_allclose(features(a,b,v),features(a*2,b*2,v),atol=2e-5)
    def test_shared_evidence_survives_every_tiled_context(self):
        torch.set_num_threads(2);m=PlumeUNet(evidence=True).eval();rng=np.random.default_rng(6);x=rng.normal(size=(17,160,160)).astype('float32');expected=1/(1+np.exp(-x[-1]))
        for window in [16,32,64,128]:np.testing.assert_allclose(predict(m,x,window,'cpu'),expected,atol=3e-6)
    def test_context_larger_than_imagery_is_rejected(self):
        with self.assertRaises(ValueError):predict(PlumeUNet(),np.zeros((17,32,32),dtype='float32'),128,'cpu')

if __name__=='__main__':unittest.main()
