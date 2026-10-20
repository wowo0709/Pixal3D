import numpy as np
import pytest

from experiments.deformnext_retargeting.surface_metrics import compare_surfaces, sample_surface


def test_translation_remains_an_error_and_distance_conventions_are_explicit():
    target=np.array([[0.,0.,0.],[1.,0.,0.]])
    prediction=target+np.array([0.,0.,.25])
    result=compare_surfaces(prediction,target,threshold=.02)
    assert result['chamfer_L1']==pytest.approx(.25)
    assert result['chamfer_squared_L2']==pytest.approx(.0625)
    assert result['surface_FScore']==0 and not result['per_output_alignment']
    assert compare_surfaces(prediction,target,threshold=.3)['surface_FScore']==1


def test_area_sampling_uses_triangle_area_and_is_reproducible():
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                       [10.,0.,0.],[12.,0.,0.],[10.,2.,0.]])
    faces=np.array([[0,1,2],[3,4,5]])
    sampled=sample_surface(vertices,faces,count=20000,seed=101)
    assert np.array_equal(sampled,sample_surface(vertices,faces,count=20000,seed=101))
    assert (sampled[:,0]>5).mean()==pytest.approx(.8,abs=.015)
    assert np.all(sampled[:,2]==0)
    with pytest.raises(ValueError):sample_surface(vertices,np.array([[0,0,0]]),seed=101)
