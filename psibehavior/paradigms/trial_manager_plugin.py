import logging
log = logging.getLogger(__name__)

import numpy as np

from atom.api import Atom, Int, Str

from psi.context.api import counterbalanced
from psiaudio.calibration import FlatCalibration
from psiaudio.stim import bandlimited_noise, apply_cos2envelope, apply_sam_envelope
from psiaudio.stim import tone


class BaseTrialManager:

    def __init__(self, controller):
        self.controller = controller
        self.context = controller.context

    def prepare_trial(self):
        raise NotImplementedError

    def start_trial(self, delay):
        raise NotImplementedError

    def end_trial(self, delay):
        raise NotImplementedError

    def trial_complete(self, response, score, info):
        raise NotImplementedError


class DummyOutput:

    def __init__(self, fs):
        self.fs = fs
        self.calibration = FlatCalibration.unity()

    def set_waveform(self, waveform):
        pass


class DummyController:

    def get_output(self, name):
        return DummyOutput(fs=100e3)


class Tinnitus2AFCManager(BaseTrialManager):

    def __init__(self, controller):
        super().__init__(controller)
        self.output = self.controller.get_output('output_1')
        self.sync_trigger = self.controller.get_output('sync_trigger_out')

        # Attributes that need to be configured by `prepare_trial`.
        self.trial_state_str = ''
        self.trial_number = 0
        self.trial_type = None
        self.prior_response = None
        self.prior_score = None

        self.stim_config = {
            'NBN': {
                'n': 4,
                'side': 1,
            },
            'SAM': {
                'n': 4,
                'side': 2,
            },
            'silence': {
                'n': 2,
                'side': 2,
            },
        }

        nbn = apply_cos2envelope(
            bandlimited_noise(
                self.output.fs, level=80, fl=4e3, fh=8e3, duration=1,
                equalize=False, calibration=self.output.calibration
            ),
            fs=self.output.fs,
            rise_time=0.25,
        )
        sam = apply_sam_envelope(nbn, fs=self.output.fs, depth=1, fm=5,
                                 delay=0, equalize=True)
        silence = np.zeros_like(nbn)

        self.waveforms = {
            'NBN': nbn,
            'SAM': sam,
            'silence': silence,
        }

        # Build the stim sequence
        stim_seq = []
        for k, v in self.stim_config.items():
            stim_seq.extend([k] * v['n'])
        self.selector = counterbalanced(stim_seq, n=20)

    def prepare_trial(self):
        repeat_mode = self.context.get_value('repeat_incorrect')
        if self.prior_response is None:
            trial_repeat = False
            self.current_trial = next(self.selector)
        elif repeat_mode == 1 and self.prior_response == 'early_np':
            trial_repeat = True
        elif repeat_mode == 2 and self.prior_score != self.controller.scores.correct:
            trial_repeat = True
        else:
            trial_repeat = False
            self.current_trial = next(self.selector)

        response_condition = self.stim_config[self.current_trial]['side']

        self.controller.trial_state_str = \
            f'Trial {self.trial_number + 1} ' \
            f'({"Repeat " if trial_repeat else ""} {self.current_trial}), ' \
            f'respond on {response_condition}'

        trial_type = f'{self.current_trial}_repeat' if trial_repeat else self.current_trial
        self.context.set_value('trial_type', trial_type)

        with self.output.engine.lock:
            self.output.set_waveform(self.waveforms[self.current_trial])

        return response_condition

    def start_trial(self, delay):
        with self.output.engine.lock:
            ts = self.controller.get_ts()
            self.output.start_waveform(ts + delay)
            self.sync_trigger.trigger(ts + delay, 0.1)
        return ts

    def end_trial(self, delay=0):
        with self.output.engine.lock:
            ts = self.controller.get_ts()
            self.output.stop_waveform(ts + delay)

    def trial_complete(self, response, score, info):
        self.prior_response = response
        self.prior_score = score
        self.trial_number += 1


if __name__ == '__main__':
    controller = DummyController()
    manager = Tinnitus2AFCStim(controller)
    print(manager.current_trial)
