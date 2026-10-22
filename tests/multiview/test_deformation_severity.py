import numpy as np
import pytest

from experiments.deformnext_retargeting.deformation_severity import surface_deformation_summary


V=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
F=np.array([[0,1,2]])


def test_no_motion_is_exact_zero_and_repeatable():
    result=surface_deformation_summary(V,F,V)
    assert result==surface_deformation_summary(V,F,V)
    assert result['exact_area_RMS_displacement']==result['surface_MC_mean_displacement']==0
    assert set(result['surface_MC_fraction_above'].values())=={0.}


def test_translation_has_constant_surface_motion_not_independent_point_distance():
    result=surface_deformation_summary(V,F,V+np.array([0.,0.,.5]))
    assert result['exact_area_RMS_displacement']==pytest.approx(.5)
    assert result['surface_MC_mean_displacement']==pytest.approx(.5)
    assert set(result['surface_MC_fraction_above'].values())=={1.}


def test_linear_displacement_has_known_exact_triangle_area_integral():
    result=surface_deformation_summary(V,F,2*V)
    # Uniform triangle E[x^2+y^2] = 1/3, not the vertex mean 2/3.
    assert result['exact_area_RMS_displacement']==pytest.approx(np.sqrt(1/3))


def test_triangle_subdivision_cannot_change_exact_area_RMS():
    vertices=np.vstack([V,(V[0]+V[1])/2,(V[1]+V[2])/2,(V[2]+V[0])/2])
    faces=np.array([[0,3,5],[3,1,4],[5,4,2],[3,4,5]])
    assert surface_deformation_summary(vertices,faces,2*vertices)['exact_area_RMS_displacement']==pytest.approx(np.sqrt(1/3))


def test_same_scale_and_translation_do_not_create_target_recentered_frame():
    original=surface_deformation_summary(V,F,V+np.array([0.,0.,.25]))
    source=V*4+np.array([7.,8.,9.]);target=(V+np.array([0.,0.,.25]))*4+np.array([7.,8.,9.])
    transformed=surface_deformation_summary(source,F,target)
    assert original['exact_area_RMS_displacement']==transformed['exact_area_RMS_displacement']
    assert transformed['surface_MC_displacement_quantiles']==original['surface_MC_displacement_quantiles']


@pytest.mark.parametrize('source,faces,target', [
    (V,F,V[:2]),(V,np.array([[0,1,3]]),V),(V,F.astype(float),V),
    (V,np.array([[0,0,0]]),V),(V,F,V+np.nan),
])
def test_invalid_correspondence_or_geometry_is_rejected(source,faces,target):
    with pytest.raises(ValueError):surface_deformation_summary(source,faces,target)
