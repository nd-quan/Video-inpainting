import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import unittest
import torch
from diffusers import DDIMScheduler
from latent_warp_diagnostics import analyze, install_capture, resize_flow


class Tests(unittest.TestCase):
    def test_bg_composite_retains_roi_and_same_metrics(self):
        from latent_warp_diagnostics import bg_only_composite
        z=torch.randn(3,4,8,8);f=torch.randn(2,2,64,64);bg=torch.ones(3,1,64,64)
        bg[:,:,:,32:]=0
        full=analyze(z,f,bg,4);only=analyze(z,f,bg,4,bg_only=True)
        for a,b in zip(full,only):
            for k in a:self.assertEqual(a[k],b[k])
            self.assertEqual(b['direct_roi_changed_elements'],0)
            self.assertEqual(b['rescaled_roi_changed_elements'],0)
        mask=torch.rand(2,1,8,8)>0.5
        result=bg_only_composite(z[:-1],z[1:],mask)
        self.assertTrue(torch.equal(result.masked_select(~mask),z[1:].masked_select(~mask)))

    def test_unchanged_trajectory(self):
        for eta in (0., 0.5):
            a,b=DDIMScheduler(),DDIMScheduler()
            a.set_timesteps(50);b.set_timesteps(50)
            x=torch.randn(3,4,8,8);y=x.clone(); saved={}
            undo=install_capture(b,{0,10,25,40,49},saved)
            ga=torch.Generator().manual_seed(42);gb=torch.Generator().manual_seed(42)
            for t in a.timesteps:
                prediction=torch.randn_like(x)
                x=a.step(prediction,t,x,eta=eta,generator=ga,return_dict=False)[0]
                y=b.step(prediction,t,y,eta=eta,generator=gb,return_dict=False)[0]
                self.assertTrue(torch.equal(x,y))
            self.assertEqual(set(saved),{0,10,25,40,49});undo()

    def test_scale_and_identity(self):
        flow=torch.ones(1,2,512,512)*8
        self.assertTrue(torch.allclose(resize_flow(flow,(64,64)),torch.ones(1,2,64,64)))
        self.assertTrue(torch.allclose(resize_flow(flow,(256,256)),torch.ones(1,2,256,256)*4))
        z=torch.randn(3,4,8,8);f=torch.zeros(2,2,64,64);bg=torch.ones(3,1,64,64)
        for s in (1,4):
            rows=analyze(z,f,bg,s)
            for r in rows:
                self.assertAlmostEqual(r['direct_relative_mse'],r['rescaled_relative_mse'],places=5)
                self.assertAlmostEqual(r['direct_warp_gain'],0,places=5)

    def test_backward_and_masks(self):
        prev=torch.randn(4,8,8);cur=torch.zeros_like(prev);cur[:,:,:-1]=prev[:,:,1:]
        z=torch.stack([prev,cur]);f=torch.zeros(1,2,8,8);f[:,0]=1
        bg=torch.ones(2,1,8,8)
        row=analyze(z,f,bg,1)[0]
        self.assertLess(row['direct_relative_mse'],1e-10)
        self.assertEqual(row['support_pixels'],56)
        bg[1]=0
        self.assertIsNone(analyze(z,f,bg,4)[0]['direct_relative_mse'])
        with self.assertRaises(AssertionError):analyze(z,f[:0],bg,4)

    def test_original_sources_and_motion_units(self):
        z=torch.randn(3,4,8,8);saved=z.clone()
        f=torch.zeros(2,2,64,64);f[:,0]=20
        bg=torch.ones(3,1,64,64)
        rows=analyze(z,f,bg,4)
        isolated=analyze(z[1:],f[1:],bg[1:],4)[0]
        self.assertEqual(rows[1],isolated)  # not propagated from pair zero
        self.assertTrue(torch.equal(z,saved))
        self.assertEqual(rows[0]['motion_bins']['motion_16_32']['motion_pixel_count'],64)
        self.assertIsNone(rows[0]['motion_bins']['motion_0_4']['direct_relative_mse'])
        self.assertLessEqual(rows[0]['common_valid_ratio'],rows[0]['primary_bg_valid_ratio'])


if __name__=='__main__':unittest.main()
