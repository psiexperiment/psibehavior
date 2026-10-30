import numpy as np
import pytest

from psibehavior.paradigms.behavior_nafc_plugin import NAFCTrialScore
from stubs import (LegacyToyManager, Subject, StubController, ToyManager,
                   run_trial)


SEQUENCE_KEYS = ('trial_type', 'trial_subtype', 'level', 'conditions',
                 'response', 'score', 'stim_start', 'stim_complete')


def run_session(manager_class, n_trials=200, N_response=1, seed=0,
                extra_prepares=None, between_prepares=None, remind=False,
                **parameters):
    # The pre-M0 implementation rewarded references only via deliver_reward,
    # which ToyManager never does for them. Match it unless told otherwise.
    parameters.setdefault('reward_correct_reference', False)
    controller = StubController(N_response=N_response, **parameters)
    manager = manager_class(controller, seed=seed)
    subject = Subject(N_response=N_response, seed=seed + 1)
    rng = np.random.default_rng(seed + 2)
    remind_rng = np.random.default_rng(seed + 3)

    def request_remind(controller):
        # The operator clicks Request Remind at random times, either between
        # trials or while a trial is playing.
        if remind_rng.uniform() < 0.1:
            controller._remind_requested = True

    hooks = {}
    if remind:
        hooks = {'before_prepare': request_remind,
                 'during_trial': request_remind}

    rows = []
    for i in range(n_trials):
        n_extra = 0 if extra_prepares is None else rng.integers(0, extra_prepares + 1)
        row = run_trial(manager, subject, n_extra, between_prepares, **hooks)
        rows.append({k: row.get(k) for k in SEQUENCE_KEYS})
    return rows, manager


@pytest.mark.parametrize('N_response', [1, 2])
@pytest.mark.parametrize('repeat_incorrect', [0, 1, 2, 3])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_matches_legacy(N_response, repeat_incorrect, seed):
    '''
    Without re-prepares, the hardened manager must produce exactly the same
    trial sequence as the pre-M0 implementation.
    '''
    expected, _ = run_session(LegacyToyManager, N_response=N_response,
                              seed=seed, repeat_incorrect=repeat_incorrect)
    actual, _ = run_session(ToyManager, N_response=N_response, seed=seed,
                            repeat_incorrect=repeat_incorrect)
    assert actual == expected
    subtypes = {r['trial_subtype'] for r in actual}
    assert {'remind', 'forced'} <= subtypes
    if repeat_incorrect:
        assert 'repeat' in subtypes


@pytest.mark.parametrize('N_response', [1, 2])
@pytest.mark.parametrize('repeat_incorrect', [0, 2])
@pytest.mark.parametrize('seed', [0, 1])
def test_matches_legacy_with_remind_requests(N_response, repeat_incorrect,
                                             seed):
    '''
    The remind flag is now cleared when the remind trial starts rather than
    when it is prepared. Without re-prepares, remind requests must still
    produce exactly the same trial sequence as the pre-M0 implementation.
    '''
    kw = dict(N_response=N_response, seed=seed,
              repeat_incorrect=repeat_incorrect, remind=True)
    expected, _ = run_session(LegacyToyManager, **kw)
    actual, _ = run_session(ToyManager, **kw)
    assert actual == expected
    # Requests outside the initial remind trials were honored.
    n_remind = sum(r['trial_subtype'] == 'remind' for r in actual[6:])
    assert n_remind > 5


@pytest.mark.parametrize('N_response', [1, 2])
@pytest.mark.parametrize('repeat_incorrect', [0, 1, 2, 3])
def test_reprepare_does_not_change_sequence(N_response, repeat_incorrect):
    '''
    Preparing a trial more than once before it starts (e.g., Apply with no
    changes, Resume) must not re-draw stimuli or double-count references.
    '''
    expected, _ = run_session(ToyManager, N_response=N_response,
                              repeat_incorrect=repeat_incorrect)
    actual, manager = run_session(ToyManager, N_response=N_response,
                                  repeat_incorrect=repeat_incorrect,
                                  extra_prepares=3)
    assert actual == expected
    # Every prepared waveform was either played or cancelled; none linger.
    for w in manager.output.waveforms:
        assert w.started or w.cancelled


def test_legacy_reprepare_is_broken():
    '''
    Sanity check that the test above would catch the legacy behavior.
    '''
    expected, _ = run_session(LegacyToyManager)
    actual, _ = run_session(LegacyToyManager, extra_prepares=3)
    assert actual != expected


def test_reuse_after_apply_rewrites_context():
    controller = StubController()
    manager = ToyManager(controller)
    manager.prepare_trial()
    stim = {k: controller.context.get_value(k)
            for k in ('trial_type', 'trial_subtype', 'level')}
    calls = manager.next_stim_calls

    # Applying changes clears values set in the context.
    controller.context.apply_changes()
    controller.context.next_setting()
    with pytest.raises(ValueError):
        controller.context.get_value('trial_type')

    manager.prepare_trial()
    assert manager.next_stim_calls == calls
    for k, v in stim.items():
        assert controller.context.get_value(k) == v
    assert len(controller.output.waveforms) == 1


def test_parameter_change_replans_and_cancels_waveform():
    controller = StubController(remind_trials=0, warmup_trials=0,
                                target_probability=1)
    manager = ToyManager(controller)
    manager.prepare_trial()
    first = controller.output.waveforms[-1]
    assert controller.context.get_value('level') in (20, 40, 60)

    controller.context.change(level_list=[80])
    controller.context.apply_changes()
    controller.context.next_setting()
    manager.prepare_trial()
    assert first.cancelled
    assert controller.context.get_value('level') == 80
    second = controller.output.waveforms[-1]
    assert second is not first and not second.cancelled

    # A behavior-only change (not a trial manager parameter) reuses the trial.
    controller.context.change(iti_duration=5)
    controller.context.apply_changes()
    controller.context.next_setting()
    manager.prepare_trial()
    assert controller.output.waveforms[-1] is second
    assert not second.cancelled


def test_started_waveform_never_cancelled_by_prepare():
    controller = StubController(remind_trials=0, warmup_trials=0,
                                target_probability=1)
    manager = ToyManager(controller)
    manager.prepare_trial()
    manager.start_trial(0)
    played = controller.output.waveforms[-1]
    stim = controller.context.get_values()
    trial_state_str = controller.trial_state_str
    calls = manager.next_stim_calls

    # Should not happen in practice, but must not touch the playing trial:
    # no new plan, waveform or context values (which would be logged with
    # the trial that is playing).
    controller.context.change(level_list=[80])
    controller.context.apply_changes()
    manager.prepare_trial()
    assert not played.cancelled
    assert manager.next_stim_calls == calls
    assert len(controller.output.waveforms) == 1
    assert controller.context.get_value('level') == stim['level']
    assert controller.trial_state_str == trial_state_str

    # Nor can a second trial be started on top of it.
    with pytest.raises(RuntimeError):
        manager.start_trial(0)

    # Invalid trials cancel the waveform that is playing.
    manager.end_trial()
    assert played.cancelled


def test_failed_replan_keeps_prepared_trial():
    '''
    If planning fails after a change, the prepared trial must not be
    cancelled: reverting the change reuses it, and it must then play.
    '''
    controller = StubController(remind_trials=0, warmup_trials=0,
                                target_probability=1)
    manager = ToyManager(controller)
    manager.prepare_trial()
    first = controller.output.waveforms[-1]
    level = controller.context.get_value('level')

    controller.context.parameters['level_list'] = [-1]
    with pytest.raises(ValueError):
        manager.prepare_trial()
    assert not first.cancelled
    assert len(controller.output.waveforms) == 1

    # Still invalid: plans again (and fails again).
    with pytest.raises(ValueError):
        manager.prepare_trial()

    # Reverted: the original trial is reused and plays.
    controller.context.parameters['level_list'] = [20, 40, 60]
    manager.prepare_trial()
    assert controller.context.get_value('level') == level
    manager.start_trial(0)
    assert first.started and not first.cancelled


def test_failed_replan_then_new_values_cancels_old_trial():
    controller = StubController(remind_trials=0, warmup_trials=0,
                                target_probability=1)
    manager = ToyManager(controller)
    manager.prepare_trial()
    first = controller.output.waveforms[-1]

    controller.context.parameters['level_list'] = [-1]
    with pytest.raises(ValueError):
        manager.prepare_trial()
    controller.context.parameters['level_list'] = [80]
    manager.prepare_trial()
    assert first.cancelled
    assert controller.context.get_value('level') == 80
    assert not controller.output.waveforms[-1].cancelled


def test_end_trial_without_started_trial_drops_prepared_trial():
    controller = StubController()
    manager = ToyManager(controller)
    manager.prepare_trial()
    first = controller.output.waveforms[-1]
    manager.end_trial()
    assert first.cancelled
    # The cancelled trial is never reused.
    manager.prepare_trial()
    second = controller.output.waveforms[-1]
    assert second is not first
    manager.start_trial(0)
    assert second.started


def test_remind_request_survives_reprepare():
    controller = StubController(remind_trials=0, warmup_trials=0,
                                target_probability=0)
    manager = ToyManager(controller)
    subject = Subject(p_early=0)

    manager.prepare_trial()
    assert controller.context.get_value('trial_type') == 'reference'

    controller._remind_requested = True
    manager.prepare_trial()
    assert controller.context.get_value('trial_subtype') == 'remind'
    assert controller.context.get_value('level') == 60

    # Re-preparing (e.g., Apply) before the trial starts keeps the request.
    manager.prepare_trial()
    assert controller._remind_requested
    assert controller.context.get_value('trial_subtype') == 'remind'

    # Cleared once the remind trial starts.
    manager.start_trial(0)
    assert not controller._remind_requested

    # A request made during the remind trial produces another remind trial.
    controller._remind_requested = True
    response, score = subject.respond('target')
    manager.trial_complete(response, score, {})
    manager.prepare_trial()
    assert controller.context.get_value('trial_subtype') == 'remind'


def test_consecutive_reference_counted_at_completion():
    controller = StubController(remind_trials=0, warmup_trials=0,
                                target_probability=0, max_reference=2)
    manager = ToyManager(controller)
    subject = Subject(p_early=0, p_correct=1)
    subtypes = []
    for i in range(4):
        row = run_trial(manager, subject, extra_prepares=5)
        subtypes.append(row['trial_subtype'])
    # max_reference=2 allows three references (legacy `>` comparison) and
    # then forces a target.
    assert subtypes == [None, None, None, 'forced']


def test_repeat_uses_completed_trial():
    '''
    Repeat mode 3 (false alarms only) must look at the trial that was played,
    not a trial that was prepared and then discarded.
    '''
    controller = StubController(remind_trials=0, warmup_trials=0,
                                target_probability=1, repeat_incorrect=3)
    manager = ToyManager(controller)

    # A target trial that is scored incorrect.
    manager.prepare_trial()
    manager.start_trial(0)
    manager.trial_complete('resp_1', NAFCTrialScore.incorrect, {})

    # Target trials are never repeated in mode 3, even if a reference trial
    # was prepared (and discarded) in between.
    controller.context.change(target_probability=0)
    controller.context.apply_changes()
    controller.context.next_setting()
    manager.prepare_trial()
    assert controller.context.get_value('trial_subtype') is None


@pytest.mark.parametrize('N_response', [1, 2])
def test_reward_conditions(N_response):
    controller = StubController(N_response=N_response, min_reward_level=30)
    manager = ToyManager(controller)
    target_easy = {'trial_type': 'target', 'level': 60}
    target_hard = {'trial_type': 'target', 'level': 20}
    reference = {'trial_type': 'reference', 'level': 0}
    port = 1 if N_response == 1 else 2

    assert manager.get_conditions(target_easy)[1] == [port]
    assert manager.get_conditions(target_hard)[1] == []

    if N_response == 1:
        # Withholding a response is correct, but there is nothing to reward.
        assert manager.get_conditions(reference)[1] == []
    else:
        # Rewarded by default.
        assert manager.get_conditions(reference)[1] == [1]

        # Unchecking the setting stops it, even when deliver_reward would
        # reward the reference stimulus (deliver_reward applies to targets
        # only).
        controller.context.parameters['reward_correct_reference'] = False
        controller.context.parameters['min_reward_level'] = 0
        assert manager.get_conditions(reference)[1] == []
        assert manager.get_conditions(target_hard)[1] == [2]


def test_prefix_passed_to_get_value():
    controller = StubController()
    controller.context.parameters['toy_level_list'] = [80]
    manager = ToyManager(controller, seed=0)
    manager.prefix = 'toy_'
    assert manager.get_value('level_list') == [80]


class KwargsManager(ToyManager):
    '''
    Adds a parameter via `get_default_parameters` (not in
    `default_parameters`) that the stimulus depends on.
    '''

    @classmethod
    def get_default_parameters(cls, offset=False, **kwargs):
        parameters = list(cls.default_parameters)
        if offset:
            parameters.append({'name': 'level_offset', 'label': 'Offset',
                               'default': 0, 'scope': 'arbitrary',
                               'group_name': 'toy'})
        return parameters

    def next_stim(self, target_probability, remind=False):
        stim = super().next_stim(target_probability, remind)
        stim['level'] += self.get_value('level_offset')
        return stim


def test_parameter_from_manager_kwargs_replans():
    controller = StubController(remind_trials=0, warmup_trials=0,
                                target_probability=1, level_list=[60],
                                level_offset=0)
    manager = KwargsManager(controller)
    manager.manager_kwargs = {'offset': True}
    manager.prepare_trial()
    assert controller.context.get_value('level') == 60

    controller.context.change(level_offset=10)
    controller.context.apply_changes()
    controller.context.next_setting()
    manager.prepare_trial()
    assert controller.context.get_value('level') == 70
    assert controller.output.waveforms[0].cancelled


def test_get_default_parameters():
    from psibehavior.paradigms.api import BaseTrialManager
    assert ToyManager.get_default_parameters() is ToyManager.default_parameters
    assert ToyManager.get_default_parameters(unused=1) is ToyManager.default_parameters
    assert BaseTrialManager.get_default_parameters() == []
