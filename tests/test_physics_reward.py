import unittest
import numpy as np
from rlscm.environment import PipeEnv, blur_misfit, image_misfit, le2024_misfit
from rlscm.reward_protocol import validate_resume

class Backend:
    def solve(self,parameters):
        image=np.zeros((256,256),np.int8)
        image[10:250,30+int(parameters[0]*10)]=1
        image[10:250,160]=1
        return {'image':image}

class PhysicsRewardTest(unittest.TestCase):
    def test_dispatch_and_no_label_reward(self):
        backend=Backend();image=backend.solve([1,2,.3])['image']
        for mode,score in [('blur',lambda a,b:blur_misfit(a,b,(4,12))),
                           ('nearest',image_misfit),('le2024',le2024_misfit)]:
            environments=[PipeEnv(backend,[target],image_score=True,score_sigmas=(4,12),
                                 misfit_mode=mode,param_weight=0) for target in
                          ({'image':image},{'image':image,'parameters':[5,3,.4]})]
            values=[]
            for env in environments:
                env.reset(seed=2026)
                observation,reward,_,_,info=env.step([.2,-.1,.1])
                self.assertAlmostEqual(info['J'],score(image,env.candidate['image']))
                self.assertAlmostEqual(reward,-info['J']-.01)
                values.append((observation,reward))
            np.testing.assert_array_equal(values[0][0]['state'],values[1][0]['state'])
            np.testing.assert_array_equal(values[0][0]['image'],values[1][0]['image'])
            self.assertEqual(values[0][1],values[1][1])

    def test_resume_guards(self):
        base=dict(reward_mode='absolute',param_weight=0.,misfit_mode='blur',score_sigmas=[4.,12.])
        validate_resume(base,base)
        for key,value in [('param_weight',2.),('misfit_mode','le2024'),
                          ('reward_mode','parameter_progress'),('score_sigmas',[12.])]:
            with self.assertRaises(ValueError):validate_resume(base,dict(base,**{key:value}))

if __name__=='__main__':unittest.main()
