'''
Stand-ins for the psiexperiment controller and output so that trial managers
and the behavior plugin can be exercised without hardware or a GUI. The
context is a real one (see `context.make_context`).
'''
from threading import RLock

import numpy as np

from context import make_context
from psibehavior.paradigms.api import NAFCTrialManager
from psibehavior.paradigms.behavior_nafc_plugin import NAFCTrialScore
from legacy_nafc import LegacyNAFCTrialManager


class FakeWaveform:

    def __init__(self, samples):
        self.samples = samples
        self.actual_ts = None
        self.started = False
        self.cancelled = False

    def start(self, ts, allow_belated=False):
        assert not self.cancelled, 'Starting a cancelled waveform'
        self.started = True
        self.actual_ts = ts

    def cancel(self):
        self.cancelled = True

    @property
    def complete(self):
        return self.started and not self.cancelled


class FakeEngine:

    def __init__(self):
        self.lock = RLock()


class FakeOutput:

    fs = 1000

    def __init__(self):
        self.engine = FakeEngine()
        self.waveforms = []

    def add_waveform(self, samples):
        waveform = FakeWaveform(samples)
        self.waveforms.append(waveform)
        return waveform


class StubController:
    '''
    Minimal controller for exercising a trial manager on its own.
    '''
    scores = NAFCTrialScore

    def __init__(self, N_response=1, manager_path='stubs.ToyManager',
                 prefix='', manager_kwargs=None, **settings):
        self.context = make_context(manager_path, N_response, prefix,
                                    manager_kwargs, **settings)
        self.N_response = N_response
        self._remind_requested = False
        self.trial_state_str = ''
        self.output = FakeOutput()
        self.ts = 0.0

    def get_ts(self):
        return self.ts

    def get_output(self, name):
        return self.output


class ToyMixin:
    '''
    Deterministic paradigm that mimics the structure of the abts managers
    (e.g., ToneDetectionManager). Louder levels are easier.
    '''

    default_parameters = [
        {'name': 'level_list', 'label': 'Levels', 'default': [20, 40, 60],
         'scope': 'arbitrary', 'group_name': 'toy'},
        {'name': 'min_reward_level', 'label': 'Min. reward level',
         'default': 30, 'scope': 'arbitrary', 'group_name': 'toy'},
        {'name': 'level', 'label': 'Level', 'type': 'Result',
         'group_name': 'toy'},
    ] + NAFCTrialManager.default_parameters

    def __init__(self, controller, prefix='', seed=0):
        super().__init__(controller, prefix)
        self.output = controller.get_output('output_1')
        self.rng = np.random.default_rng(seed)
        self.next_stim_calls = 0

    def next_stim(self, target_probability, remind=False):
        self.next_stim_calls += 1
        levels = self.get_value('level_list')
        if self.rng.uniform() >= target_probability:
            return {'level': 0, 'trial_type': 'reference'}
        if remind:
            level = max(levels)
        else:
            level = levels[self.rng.integers(len(levels))]
        return {'level': level, 'trial_type': 'target'}

    def deliver_reward(self, stim):
        return stim['level'] >= self.get_value('min_reward_level')

    def get_stim_str(self, stim):
        return f'{stim["level"]} dB'

    def stim_waveform(self, stim):
        # Lets tests make planning fail (e.g., level_list=[-1]).
        if stim['level'] < 0:
            raise ValueError(f'Invalid level {stim["level"]}')
        return np.full(10, float(stim['level']))


class ToyManager(ToyMixin, NAFCTrialManager):
    pass


class LegacyToyManager(ToyMixin, LegacyNAFCTrialManager):
    pass


class Subject:
    '''
    Scripted responses. `respond(trial_type)` returns (response, score) for a
    single-port (go/nogo) or two-port task.
    '''

    def __init__(self, N_response=1, seed=0, p_correct=0.7, p_early=0.1):
        self.N_response = N_response
        self.rng = np.random.default_rng(seed)
        self.p_correct = p_correct
        self.p_early = p_early

    def respond(self, trial_type):
        if self.rng.uniform() < self.p_early:
            return 'early_np', NAFCTrialScore.invalid
        correct = self.rng.uniform() < self.p_correct
        if self.N_response == 1:
            if trial_type == 'target':
                if correct:
                    return 'resp_1', NAFCTrialScore.correct
                return 'no_response', NAFCTrialScore.invalid
            if correct:
                return 'no_response', NAFCTrialScore.correct
            return 'resp_1', NAFCTrialScore.incorrect
        correct_side = 1 if trial_type == 'reference' else 2
        side = correct_side if correct else 3 - correct_side
        score = NAFCTrialScore.correct if correct else NAFCTrialScore.incorrect
        return f'resp_{side}', score


def run_trial(manager, subject, extra_prepares=0, between_prepares=None,
              before_prepare=None, during_trial=None):
    '''
    Run one trial through the manager the way BehaviorPlugin does.

    `before_prepare` and `during_trial` are called with the controller before
    the trial is prepared and once it has started (e.g., to request a remind
    trial).

    Returns the logged row.
    '''
    controller = manager.controller
    if before_prepare is not None:
        before_prepare(controller)
    conditions = manager.prepare_trial()
    for i in range(extra_prepares):
        if between_prepares is not None:
            between_prepares(i)
        conditions = manager.prepare_trial()
    trial_type = controller.context.get_value('trial_type')
    manager.start_trial(0.5)
    if during_trial is not None:
        during_trial(controller)
    response, score = subject.respond(trial_type)
    info = {'response': response, 'score': score.name}
    if score == NAFCTrialScore.invalid:
        manager.end_trial()
    manager.trial_complete(response, score, info)
    row = {**info, **controller.context.get_values()}
    row['conditions'] = conditions
    controller.ts += 5
    return row
