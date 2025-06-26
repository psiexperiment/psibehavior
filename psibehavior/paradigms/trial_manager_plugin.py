import numpy as np

from psi.context.api import counterbalanced
from psiaudio.calibration import FlatCalibration
from psiaudio.stim import bandlimited_noise, apply_cos2envelope, apply_sam_envelope
from psiaudio.stim import tone


class BaseTrialManager:

    def __init__(self, controller):
        self.controller = controller

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
        self.stim_config = {
            'NBN': {
                'n': 4,
                'side': 1,
            },
            #'SAM': {
            #    'n': 4,
            #    'side': 2,
            #},
            #'silence': {
            #    'n': 2,
            #    'side': 1,
            #},
        }

        #nbn = apply_cos2envelope(
        #    bandlimited_noise(
        #        self.output.fs, level=80, fl=4e3, fh=8e3, duration=1,
        #        equalize=False, calibration=self.output.calibration
        #    ),
        #    fs=self.output.fs,
        #    rise_time=0.25,
        #)
        nbn = apply_cos2envelope(
            tone(
                self.output.fs, level=80, frequency=4e3, duration=1,
                calibration=self.output.calibration
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
        self.selector = counterbalanced(stim_seq, n=20, c=1)
        self.current_trial = next(self.selector)

    def prepare_trial(self):
        # Get waveform
        with self.output.engine.lock:
            self.output.set_waveform(self.waveforms[self.current_trial])
        self.controller.side = self.stim_config[self.current_trial]['side']

    def start_trial(self, delay):
        with self.output.engine.lock:
            ts = self.controller.get_ts()
            self.output.start_waveform(ts + delay)
            self.sync_trigger.trigger(ts + delay, 0.1)
        return ts

    def end_trial(self, delay=0):
        with self.output.engine.lock:
            ts = self.get_ts()
            self.output.stop_waveform(ts + delay)

    def trial_complete(self, response, score, info):
        self.current_trial = next(self.selector)


if __name__ == '__main__':
    controller = DummyController()
    manager = Tinnitus2AFCStim(controller)
    print(manager.current_trial)
