'''
Tests for `SDTAnalysisPlugin`: d' against the stimulus-specific or overall
false alarm rate, and the reaction times shown by `ReactionTimePlotManifest`.
'''
import enaml
import numpy as np
import pytest
from scipy.stats import norm

with enaml.imports():
    from psibehavior.paradigms.sdt_analysis import (
        SDTAnalysisPlugin, target_results,
    )


def trial(trial_type, correct, reaction_time=np.nan, **values):
    return {
        'trial_type': trial_type, 'trial_subtype': None, 'correct': correct,
        'reaction_time': reaction_time, **values,
    }


def analyze(data, grouping, fa_mode='specific'):
    plugin = SDTAnalysisPlugin(fa_mode=fa_mode)
    plugin.data = data
    plugin.grouping = grouping
    return plugin


#: Tone detection: reference trials play no tone, so they have no frequency
#: or level.
TONE_DETECTION = [
    trial('target', True, 0.5, level=40),
    trial('target', True, 0.7, level=40),
    # A miss: not a response, so not in the mean reaction time.
    trial('target', False, 1.5, level=40),
    trial('target', False, level=40),
    trial('target', True, 0.3, level=60),
    # One false alarm in four reference trials.
    trial('reference', False, 2.0, level=None),
    trial('reference', True, level=None),
    trial('reference', True, level=None),
    trial('reference', True, level=None),
]


def test_overall_fa_rate():
    plugin = analyze(TONE_DETECTION, ['level', 'trial_type'], 'overall')
    assert plugin.fa == 0.25
    targets = target_results(plugin.results).set_index('level')
    # Hit rates are clipped to 0.05-0.95, as is the FA rate.
    fa_z = norm.ppf(0.25)
    assert targets.loc[40, 'z_score_norm'] == pytest.approx(norm.ppf(0.5) - fa_z)
    assert targets.loc[60, 'z_score_norm'] == pytest.approx(norm.ppf(0.95) - fa_z)


def test_specific_fa_rate_cannot_match_tone_detection():
    # Why tone detection uses 'overall': its reference trials are dropped
    # from the grouping, so there is nothing to compare each level with.
    plugin = analyze(TONE_DETECTION, ['level', 'trial_type'])
    assert target_results(plugin.results)['z_score_norm'].isna().all()


def test_specific_fa_rate():
    # Gap detection: reference trials (no gap) have a frequency, and each
    # frequency's gaps are compared with its own FA rate.
    data = [
        trial('target', True, gap=0.032, frequency=4000),
        trial('target', False, gap=0.032, frequency=4000),
        trial('reference', False, gap=0, frequency=4000),
        trial('reference', True, gap=0, frequency=4000),
        trial('reference', True, gap=0, frequency=4000),
        trial('reference', True, gap=0, frequency=4000),
        trial('target', True, gap=0.032, frequency=8000),
        trial('target', True, gap=0.032, frequency=8000),
        trial('reference', False, gap=0, frequency=8000),
        trial('reference', True, gap=0, frequency=8000),
    ]
    plugin = analyze(data, ['frequency', 'gap', 'trial_type'])
    targets = target_results(plugin.results).set_index('frequency')
    assert targets.loc[4000, 'z_score_norm'] == \
        pytest.approx(norm.ppf(0.5) - norm.ppf(0.25))
    assert targets.loc[8000, 'z_score_norm'] == \
        pytest.approx(norm.ppf(0.95) - norm.ppf(0.5))


def test_reaction_time_of_hits():
    plugin = analyze(TONE_DETECTION, ['level', 'trial_type'], 'overall')
    targets = target_results(plugin.results).set_index('level')
    assert targets.loc[40, 'reaction_time'] == pytest.approx(0.6)
    assert targets.loc[60, 'reaction_time'] == pytest.approx(0.3)
    assert target_results(None) is None
