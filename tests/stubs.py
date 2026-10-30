'''
Stand-ins for the psiexperiment context, controller and output so that trial
managers and the behavior plugin can be exercised without hardware or a GUI.
'''
from threading import RLock

import numpy as np

from psibehavior.paradigms.api import NAFCTrialManager
from psibehavior.paradigms.behavior_nafc_plugin import NAFCTrialScore
from legacy_nafc import LegacyNAFCTrialManager


DEFAULT_PARAMETERS = {
    # Trial manager
    'target_probability': 0.5,
    'max_reference': 3,
    'remind_trials': 2,
    'warmup_trials': 4,
    'level_list': [20, 40, 60],
    'min_reward_level': 30,
    # Behavior plugin
    'repeat_incorrect': 0,
    'reward_correct_reference': False,
    'training_mode': False,
    'manual_control': False,
    'continuous_np': False,
    'np_duration': 0.25,
    'target_delay': 0.5,
    'hold_duration': 0.0,
    'response_delay': 0,
    'response_duration': 3.0,
    'iti_duration': 0.1,
    'to_duration': 1.0,
}


class Invalid:
    '''
    A parameter value that fails validation (e.g., an expression with a typo).
    '''

    def __init__(self, message='invalid expression'):
        self.message = message


class ContextNamespace:
    '''
    Mimics the parts of the psiexperiment context plugin used by trial
    managers.

    Parameters evaluate to their current value. Values set via `set_value(s)`
    last until `next_setting` is called (as happens when changes are applied).
    Parameter changes made via `change` only take effect on `apply_changes`.
    '''

    def __init__(self, **parameters):
        self.parameters = {**DEFAULT_PARAMETERS, **parameters}
        self.locals = {}
        self.pending = {}
        self.applied = 0

    def get_value(self, name):
        if name in self.locals:
            return self.locals[name]
        if name in self.parameters:
            return self.parameters[name]
        raise ValueError(f'Name {name} not defined')

    def get_values(self):
        return {**self.parameters, **self.locals}

    def set_value(self, name, value):
        self.locals[name] = value

    def set_values(self, values):
        self.locals.update(values)

    def change(self, **parameters):
        self.pending.update(parameters)

    def validate_expressions(self):
        # Like psiexperiment, this checks the changes that have not been
        # applied yet.
        for name, value in self.pending.items():
            if isinstance(value, Invalid):
                raise ValueError(f'"{name}": {value.message}')

    def apply_changes(self):
        self.validate_expressions()
        self.parameters.update(self.pending)
        self.pending = {}
        self.applied += 1

    def next_setting(self, selector=None):
        self.locals = {}


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

    def __init__(self, N_response=1, **parameters):
        self.context = ContextNamespace(**parameters)
        self.N_response = N_response
        self._remind_requested = False
        self.trial_state_str = ''
        self.output = FakeOutput()
        self.ts = 0.0
        self.warnings = []

    def get_ts(self):
        return self.ts

    def report_warning(self, title, message):
        self.warnings.append(title)

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
