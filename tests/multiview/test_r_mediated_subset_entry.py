import sys

import pytest

from experiments.deformnext_retargeting.evaluate_r_mediated_subset import main


@pytest.mark.parametrize('changed, message', [
    (['--require-native-identity'], 'completed step112'),
    (['--step', '96'], 'completed step112'),
    (['--panel-size', '32'], 'frozen72 panel'),
])
def test_subset_rejects_wrong_epoch_or_panel_before_loading_model(monkeypatch, changed, message):
    args = ['entry', '--pilot-variant', 'R-mediated-rawTopK-NoDQ',
            '--appearance-mode', 'raw-topk-plus-feature', '--residual-logit-bound', '1',
            '--expected-assets', '72', '--step', '112', '--panel-size', '72', '--stage', 'ss64']
    if len(changed) == 1:
        args += changed
    else:
        args[args.index(changed[0])+1] = changed[1]
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(ValueError, match=message):
        main()
