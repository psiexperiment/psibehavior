import logging
log = logging.getLogger(__name__)

import numpy as np

from atom.api import Atom, Int, Str

from psi.context.api import counterbalanced, shuffled_set
from psiaudio.stim import bandlimited_noise, bandlimited_fir_noise, apply_cos2envelope, apply_sam_envelope
from psiaudio.stim import tone
from psiaudio.util import octave_band_freqs, octave_space


class BaseTrialManager:

    #: List of parameters to add to GUI. Eliminates the need for a custom
    #: `TrialManagerManifest` class if you only want to add new parameters to
    #: the GUI.
    default_parameters = []

    def __init__(self, controller):
        self.controller = controller
        self.context = controller.context
        self.trial_number = 0

    def prepare_trial(self):
        raise NotImplementedError

    def start_trial(self, delay):
        raise NotImplementedError

    def end_trial(self, delay):
        raise NotImplementedError

    def trial_complete(self, response, score, info):
        raise NotImplementedError


class ParamTrialManager(BaseTrialManager):
    '''
    Allows users to specify list of parameters to add to GUI without creating a
    custom manifest.
    '''


#class RepeatTrialManager(BaseTrialManager):
#
#    def __init__(self, manager):
#        super().__init__(controller)
#        self.trial_type = None
#        self.prior_response = None
#        self.prior_score = None
#
#
#class HyperacusisGoNogoManager(BaseTrialManager):
#
#    def __init__(self, manager):
#        super().__init__(controller)
#        self.output = self.controller.get_output('output_1')
#        self.sync_trigger = self.controller.get_output('sync_trigger_out')
#
#        # Attributes that need to be configured by `prepare_trial`.
#        self.trial_type = None
#        self.prior_response = None
#        self.prior_score = None


TINNITUS_FREQUENCIES = octave_space(2, 32, 0.5) * 1e3
TINNITUS_CHOICES = {f'{f*1e-3:.1f}': f for f in TINNITUS_FREQUENCIES}


class Tinnitus2AFCManager(BaseTrialManager):

    default_parameters = [
        {
            'name': 'frequencies',
            'label': 'Frequencies (kHz)',
            'group_name': 'Tinnitus 2AFC',
            'default': ['8.0'],
            'scope': 'arbitrary',
            'type': 'MultiSelectParameter',
            'choices': TINNITUS_CHOICES,
            'quote_values': False,
        },
        {
            'name': 'stimuli',
            'label': 'Stimuli',
            'group_name': 'Tinnitus 2AFC',
            'default': ['NBN', 'silence', 'SAM'],
            'scope': 'arbitrary',
            'type': 'MultiSelectParameter',
            'quote_values': True,
            'choices': {
                'NBN': 'NBN',
                'silence': 'silence',
                'SAM': 'SAM',
            },
        },
        {
            'name': 'reward_silent',
            'label': 'Reward silent trials?',
            'group_name': 'Tinnitus 2AFC',
            'default': True,
            'scope': 'arbitrary',
            'type': 'BoolParameter',
        },
        {
            'name': 'reward_rate',
            'label': 'Reward Rate (frac.)',
            'group_name': 'Tinnitus 2AFC',
            'default': 1,
            'scope': 'arbitrary',
        },

    ]

    def __init__(self, controller):
        super().__init__(controller)
        self.output = self.controller.get_output('output_1')
        self.sync_trigger = self.controller.get_output('sync_trigger_out')

        # Attributes that need to be configured by `prepare_trial`.
        self.trial_state_str = ''
        self.prior_response = None
        self.stim = None
        self.freq = None

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

        self.waveforms = {}

        for i in range(20):
            # Generate multiple waveforms to minimize reliance on seed.

            self.waveforms['SAM', i] = apply_sam_envelope(
                apply_cos2envelope(
                    bandlimited_noise(
                        self.output.fs,
                        level=80,
                        fl=2e3,
                        fh=32e3,
                        filter_rolloff=0.1,
                        duration=1,
                        equalize=True,
                        calibration=self.output.calibration,
                        seed=i,
                    ),
                    fs=self.output.fs,
                    rise_time=0.25,
                ),
                fs=self.output.fs,
                depth=1,
                fm=5,
                delay=0,
                equalize=True,
            )
            for frequency in TINNITUS_FREQUENCIES:
                fl, fh = octave_band_freqs(frequency, 1/8)
                self.waveforms['NBN', i, frequency] = apply_cos2envelope(
                    bandlimited_noise(
                        self.output.fs,
                        level=80,
                        fl=fl,
                        fh=fh,
                        filter_rolloff=0.25,
                        duration=1,
                        equalize=True,
                        calibration=self.output.calibration,
                        seed=i,
                    ),
                    fs=self.output.fs,
                    rise_time=0.25,
                )

        self.waveforms['silence'] = np.zeros_like(self.waveforms['SAM', 0])

    def advance_stim(self):
        stim = self.context.get_value('stimuli')
        freq = self.context.get_value('frequencies')

        if self.stim != stim:
            # Stim sequence has not been initialized or list of stimuli to test
            # has changed.
            log.info('Updating stimlus sequence')
            stim_seq = []
            for s in stim:
                stim_seq.extend([s] * self.stim_config[s]['n'])
            self.stim_selector = counterbalanced(stim_seq, n=20)
            self.stim = stim

        if self.freq != freq:
            # Frequency sequence has not been initialized or list of
            # frequencies to test has changed.
            log.info('Updating frequency sequence')
            self.freq_selector = shuffled_set(freq)
            self.freq = freq

        # Wrap in `str` because something is causing it to return a numpy dtype
        self.current_trial = next(self.stim_selector)
        self.current_seed = np.random.randint(0, 20)
        if self.current_trial == 'NBN':
            self.current_freq = next(self.freq_selector)

    def prepare_trial(self):
        repeat_mode = self.context.get_value('repeat_incorrect')
        if self.prior_response is None:
            trial_repeat = False
            self.advance_stim()
        elif repeat_mode == 1 and self.prior_response == 'early_np':
            trial_repeat = True
        elif repeat_mode == 2 and self.prior_score != self.controller.scores.correct:
            trial_repeat = True
        else:
            trial_repeat = False
            self.advance_stim()

        response_condition = self.stim_config[self.current_trial]['side']

        # Prepare a string representation for GUI
        stim_info = self.current_trial
        if self.current_trial == 'NBN':
            stim_info = f'{stim_info} {self.current_freq*1e-3:.1f} kHz'
        if self.current_trial in ('NBN', 'SAM'):
            stim_info = f'{stim_info} seed={self.current_seed}'
        self.controller.trial_state_str = \
            f'Trial {self.trial_number + 1}, ' \
            f'{"Repeat " if trial_repeat else ""}{stim_info}, ' \
            f'respond on {response_condition}'

        trial_type = f'{self.current_trial}_repeat' if trial_repeat else self.current_trial
        self.context.set_value('trial_type', trial_type)

        with self.output.engine.lock:
            if self.current_trial == 'NBN':
                waveform = self.waveforms['NBN', self.current_seed, self.current_freq]
            elif self.current_trial == 'SAM':
                waveform = self.waveforms['SAM', self.current_seed]
            else:
                waveform = self.waveforms[self.current_trial]
            self.output.set_waveform(waveform)

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
