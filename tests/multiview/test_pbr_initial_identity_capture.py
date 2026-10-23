from experiments.deformnext_retargeting.capture_pbr_initial_identity import differences


def test_reports_changed_input_identity_separately_from_render_metric_roundoff():
    fields=['sample_id','noise','prediction']
    reference={'per_asset':[dict(sample_id='a',noise='same',prediction='prior',metrics={'PSNR':20.,'SSIM':.9}),
                            dict(sample_id='b',noise='same',prediction='same',metrics={'PSNR':30.,'SSIM':.95})]}
    actual={'quick_per_asset':[dict(sample_id='b',noise='same',prediction='same',metrics={'PSNR':30.0001,'SSIM':.95}),
                              dict(sample_id='a',noise='same',prediction='actual',metrics={'PSNR':19.,'SSIM':.8})]}
    rows=differences(actual,reference,fields)
    assert rows[0]['identity_mismatches']=={}
    assert set(rows[0]['metric_mismatches'])=={'PSNR'}
    assert set(rows[1]['identity_mismatches'])=={'prediction'}
    assert rows[1]['metric_mismatches']['PSNR']['delta']==-1.
