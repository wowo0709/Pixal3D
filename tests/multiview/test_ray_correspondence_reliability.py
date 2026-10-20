import numpy as np

from experiments.deformnext_retargeting.ray_correspondence_reliability import (
    inference_features, binary_metrics, calibration_threshold)


def test_features_distinguish_ambiguity_without_labels_and_preserve_empty_rows():
    rng = np.random.default_rng(5)
    native, context = rng.normal(size=(3, 1024)), rng.normal(size=(3, 1024))
    valid = np.ones((3, 64), bool)
    valid[2] = False
    prior = np.zeros((3, 64), np.float32)
    prior[0, 0] = 20
    addresses = np.tile(np.arange(64), (3, 1))
    with np.errstate(invalid='raise', divide='raise'):
        actual = inference_features(native, context, prior, valid, addresses)
    assert actual.shape == (3, 2060) and np.isfinite(actual).all()
    assert actual[0, 2051] < actual[1, 2051]  # entropy
    assert actual[0, 2052] > actual[1, 2052]  # maximum probability
    assert actual[2, 2054] == 0  # candidate count
    empty = inference_features(native[:0], context[:0], prior[:0], valid[:0], addresses[:0])
    assert empty.shape == (0, 2060)


def test_binary_metrics_handle_ties_and_calibration_threshold():
    labels = np.array([False, True, False, True])
    constant = binary_metrics(labels, np.full(4, .5), .5)
    assert constant['AUROC'] == .5 and constant['Brier'] == .25
    good = np.array([.1, .8, .2, .9])
    assert binary_metrics(labels, good, .5)['AUROC'] == 1.
    threshold = calibration_threshold(labels, good)
    assert binary_metrics(labels, good, threshold)['balanced_accuracy'] == 1.
