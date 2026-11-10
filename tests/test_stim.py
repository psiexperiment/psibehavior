'''
Tests for the continuous background stimuli (`psibehavior.paradigms.stim`).
'''
from types import SimpleNamespace

import numpy as np
import pytest

from psiaudio.calibration import FlatCalibration
from psiaudio.util import db

from psibehavior.paradigms.stim import BandlimitedFIRNoise


def make_noise(level):
    values = {'masker_fl': 0.5, 'masker_fh': 8, 'masker_level': level}
    output = SimpleNamespace(
        fs=48000, callback=None, active=False,
        # 1 Vrms is 94 dB SPL.
        calibration=FlatCalibration.from_spl(94),
    )
    controller = SimpleNamespace(
        context=SimpleNamespace(get_value=lambda name: values[name]),
        get_output=lambda name: output,
    )
    noise = BandlimitedFIRNoise(controller, ['continuous_output_1'])
    return noise, values, output


def level_of(samples):
    # dB SPL, given the calibration in `make_noise`.
    return db(np.std(samples)) + 94


def test_background_noise_plays_at_its_level():
    # The band is at the level it is labelled with overall, not quieter by
    # the fraction of the spectrum it covers.
    noise, values, output = make_noise(74)
    assert output.active
    assert output.callback == noise.next
    noise.next(48000, 'continuous_output_1')  # Let the filter settle.
    samples = noise.next(96000, 'continuous_output_1')
    assert level_of(samples) == pytest.approx(74, abs=1)


def test_background_noise_level_changes_during_session():
    # The level can be changed at any time. This used to raise (the noise
    # factory has no `update_level`).
    noise, values, output = make_noise(74)
    noise.next(48000, 'continuous_output_1')
    values['masker_level'] = 54
    samples = noise.next(96000, 'continuous_output_1')
    assert level_of(samples) == pytest.approx(54, abs=1)

