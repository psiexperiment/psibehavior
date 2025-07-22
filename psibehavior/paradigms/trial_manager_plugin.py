import logging
log = logging.getLogger(__name__)

import itertools

from atom.api import Atom, Int, Str
import numpy as np

from psi.context.api import counterbalanced, shuffled_set
from psiaudio.stim import bandlimited_noise, bandlimited_fir_noise, apply_cos2envelope, apply_sam_envelope
from psiaudio.stim import silence, ramped_tone
from psiaudio.util import octave_band_freqs, octave_space


class BaseContinuousStimManager:

    #: List of parameters to add to GUI. Eliminates the need for a custom
    #: `TrialManagerManifest` class if you only want to add new parameters to
    #: the GUI.
    default_parameters = []

    #: Default group name for each parameter in `default_parameters`. If None,
    #: `group_name`, must be provided for each parameter.
    default_group_name = None

    def __init__(self, controller):
        self.controller = controller
        self.context = controller.context

    def initialize(self):
        raise NotImplementedError

    def next(self, samples, channel):
        raise NotImplementedError


class FIRBandlimitedNoise(BaseContinuousStimManager):

    def __init__(self, controller):
        super().__init__(controller)


    def next(self, samples, channel):
        return np.zeros(samples)


class BaseTrialManager:

    #: List of parameters to add to GUI. Eliminates the need for a custom
    #: `TrialManagerManifest` class if you only want to add new parameters to
    #: the GUI.
    default_parameters = []

    #: Default group name for each parameter in `default_parameters`. If None,
    #: `group_name`, must be provided for each parameter.
    default_group_name = None

    def __init__(self, controller):
        self.controller = controller
        self.context = controller.context
        self.trial_number = 0

    def prepare_trial(self):
        '''
        Subclasses are responsible for setting 'trial_type' and 'trial_subtype'
        in the context, e.g.:

            self.context.set_value('trial_type', 'nogo')
            self.context.set_value('trial_subtype', 'repeat')

        Trial type will be specific to the paradigm and should be coded
        appropriately. Trial subtype indicates if it is a special class (e.g.,
        'repeat', 'forced', 'remind', 'warmup') that may need to be excluded
        from analysis.

        Returns
        -------
        int : correct response port
            If greater than 0, indicates response port that is rewarded
            (response ports are numbered starting at 1). If 0, this indicates
            this is a nogo trial (i.e., no reward). If -1, indicates any
            response port recieves a reward. For go/nogo experiments, there is
            only one response port which will be numbered 1.
        '''
        raise NotImplementedError

    def start_trial(self, delay):
        raise NotImplementedError

    def end_trial(self, delay):
        raise NotImplementedError

    def trial_complete(self, response, score, info):
        raise NotImplementedError


TINNITUS_FREQUENCIES = octave_space(2, 32, 0.5) * 1e3
TINNITUS_CHOICES = {f'{f*1e-3:.1f}': f for f in TINNITUS_FREQUENCIES}


class Tinnitus2AFCManager(BaseTrialManager):

    default_group_name = 'Tinnitus 2AFC'

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
            'name': 'noise_seed',
            'label': 'Noise Seed',
            'type': 'Result',
        },
        {
            'name': 'nbn_frequency',
            'label': 'NBN frequency (Hz)',
            'type': 'Result',
        },
        {
            'name': 'stimuli',
            'label': 'Stimuli',
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
            'default': True,
            'scope': 'arbitrary',
            'type': 'BoolParameter',
        },
        {
            'name': 'reward_rate',
            'label': 'Reward Rate (frac.)',
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
                'n': 5,
                'side': 1,
            },
            'SAM': {
                'n': 3,
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
        if self.current_trial in ('NBN', 'SAM'):
            self.current_seed = np.random.randint(0, 20)
        else:
            self.current_seed = None
        if self.current_trial == 'NBN':
            self.current_freq = next(self.freq_selector)
        else:
            self.current_freq = None

    def prepare_trial(self):
        repeat_mode = self.context.get_value('repeat_incorrect')
        if self.prior_response is None:
            trial_subtype = None
            self.advance_stim()
        elif repeat_mode == 1 and self.prior_response == 'early_np':
            trial_subtype = 'repeat'
        elif repeat_mode == 2 and self.prior_score != self.controller.scores.correct:
            trial_subtype = 'repeat'
        else:
            trial_subtype = None
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
            f'{"Repeat " if trial_subtype is not None else ""}{stim_info}, ' \
            f'respond on {response_condition}'

        self.context.set_value('trial_type', self.current_trial)
        self.context.set_value('trial_subtype', trial_subtype)
        self.context.set_value('noise_seed', self.current_seed)
        self.context.set_value('nbn_frequency', self.current_freq)

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


GONOGO_PARAMETERS = [
    {
        'name': 'go_probability',
        'label': 'Go probability (frac.)',
        'compact_label': 'Pr',
        'default': 0.5,
        'scope': 'arbitrary',
    },
    {
        'name': 'max_nogo',
        'label': 'Max. consecutive nogo trials',
        'compact_label': 'Max. NG',
        'default': 5,
        'scope': 'arbitrary',
    },
    {
        'name': 'remind_trials',
        'label': 'Remind trials',
        'compact_label': 'N remind',
        'scope': 'experiment',
        'default': 10,
    },
    {
        'name': 'warmup_trials',
        'label': 'Warmup trials',
        'compact_label': 'N warmup',
        'scope': 'experiment',
        'default': 20,
    },
]


class GoNogoTrialManager(BaseTrialManager):

    default_parameters = GONOGO_PARAMETERS

    def __init__(self, controller):
        super().__init__(controller)
        # Attributes used to track state of trials
        self.prior_info = None, None
        self.trial_type = None
        self.trial_subtype = None
        self.rng = np.random.default_rng()
        self.consecutive_nogo = 0
        self.current_stim = {}

    def next_ttype(self):
        '''
        Determine next trial type.
        '''
        n_remind = self.context.get_value('remind_trials')
        n_warmup = self.context.get_value('warmup_trials')
        go_probability = self.context.get_value('go_probability')
        repeat_mode = self.context.get_value('repeat_incorrect')
        max_nogo = self.context.get_value('max_nogo')

        if self.trial_number < n_remind:
            return ('go', 'remind')

        if self.controller._remind_requested:
            return ('go', 'remind')
            self.controller._remind_requested = False

        if self.trial_number < (n_remind + n_warmup):
            return ('go', 'remind') \
                if self.rng.uniform() <= go_probability else ('nogo', None)

        if self.consecutive_nogo >= max_nogo:
            return ('go', 'forced')

        if repeat_mode != 0:
            ttype, resp, score = self.prior_info
            if repeat_mode == 1 and resp == 'early_np':
                # early withdraw only
                return (ttype, 'repeat')
            if repeat_mode == 2 and score != self.controller.scores.correct:
                # all incorrect trials
                return (ttype, 'repeat')
            if repeat_mode == 3 and ttype == 'nogo' and \
                    score != self.controller.scores.correct:
                # Only FA trials
                return (ttype, 'repeat')

        return ('go', None) \
            if self.rng.uniform() <= go_probability else ('nogo', None)

    def prepare_trial(self):
        self.trial_type, self.trial_subtype = self.next_ttype()
        self.current_stim = self.next_stim(self.trial_type, self.trial_subtype)

        # Update context with some information about trial and stim
        self.context.set_value('trial_type', self.trial_type)
        self.context.set_value('trial_subtype', self.trial_subtype)
        self.context.set_values(self.current_stim)

        # Prepare a string representation for GUI
        stim_info = self.stim_info(self.current_stim)
        ttype_info = f'Trial {self.trial_number + 1}: {self.trial_type}'
        if self.trial_subtype is not None:
            ttype_info = f'{ttype_info} ({self.trial_subtype})'
        self.controller.trial_state_str = f'{ttype_info}{stim_info}'

        waveform = self.stim_waveform(self.current_stim)
        with self.output.engine.lock:
            self.output.set_waveform(waveform)

        return 0 if self.trial_type.startswith('nogo') else 1

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
        self.prior_info = self.trial_type, response, score
        self.trial_number += 1
        if self.trial_type.startswith('nogo'):
            self.consecutive_nogo += 1
        else:
            self.consecutive_nogo = 0

    def next_stim(self, trial_type):
        raise NotImplementedError

    def stim_info(self, stim):
        raise NotImplementedError

    def stim_waveform(self, stim):
        raise NotImplementedError


class HyperacusisGoNogoManager(GoNogoTrialManager):

    default_group_name = 'Hyperacusis'

    default_parameters = [
        {
            'name': 'frequency_list',
            'label': 'Frequencies (kHz)',
            'default': ['8.0'],
            'scope': 'arbitrary',
            'type': 'MultiSelectParameter',
            'choices': TINNITUS_CHOICES,
            'quote_values': False,
            'button_width': 40,
            'n_cols': 4,
        },
        {
            'name': 'level_list',
            'label': 'Levels (dB SPL)',
            'default': ['80'],
            'scope': 'arbitrary',
            'type': 'MultiSelectParameter',
            'choices': {str(l): l for l in np.arange(0, 81, 10).astype('i')},
            'quote_values': False,
            'button_width': 40,
            'n_cols': 4,
        },
        {
            'name': 'frequency',
            'label': 'Frequency (kHz)',
            'type': 'Result',
        },
        {
            'name': 'level',
            'label': 'Level (dB SPL)',
            'type': 'Result',
        },
    ] + GoNogoTrialManager.default_parameters

    def __init__(self, controller):
        super().__init__(controller)
        self.output = self.controller.get_output('output_1')
        self.sync_trigger = self.controller.get_output('sync_trigger_out')
        # Attributes that need to be configured by `prepare_trial`.
        self.frequencies = None
        self.levels = None

    def next_stim(self, trial_type, trial_subtype):
        # First, check to see if any context values have changed and, if so,
        # update the selector.
        frequencies = self.context.get_value('frequency_list')
        levels = self.context.get_value('level_list')

        if (self.frequencies != frequencies) or (self.levels != levels):
            # Regenerate stim sequence
            sequence = list(itertools.product(frequencies, levels))
            self.stim_selector = shuffled_set(sequence)
            self.frequencies = frequencies
            self.levels = levels

        if trial_subtype == 'remind':
            # Should be an easy go trial. Randomly pick from selected
            # frequencies and use max level configured.
           return {
               'frequency': self.rng.choice(self.frequencies),
               'level': max(self.levels),
           }

        if trial_subtype == 'repeat':
            return self.current_stim

        if trial_type == 'nogo':
            return {
                'frequency': None,
                'level': None,
            }

        if trial_type == 'go':
            freq, level = next(self.stim_selector)
            return {
                'frequency': freq,
                'level': level,
            }


    def stim_info(self, stim):
        if stim['frequency'] is None:
            return ''
        return f' {stim["frequency"]*1e-3:.1f} Hz, {stim["level"]:.0f} dB SPL'

    def stim_waveform(self, stim):
        if stim['frequency'] is None:
            return silence(fs=self.output.fs, duration=1)
        return ramped_tone(
            **stim,
            fs=self.output.fs,
            calibration=self.output.calibration,
            duration=1,
            window='cosine-squared',
            rise_time=25e-3,
        )


class WavGoNogoManager(GoNogoTrialManager):

    default_group_name = 'Stimuli'

    default_parameters = [
        {
            'name': 'level',
            'label': 'Level (dB SPL)',
            'default': 80,
        },
    ] + GoNogoTrialManager.default_parameters

    def __init__(self, controller):
        super().__init__(controller)
        self.output = self.controller.get_output('output_1')
        self.sync_trigger = self.controller.get_output('sync_trigger_out')

    def next_stim(self, trial_type):
        if (self.frequencies != frequencies) or (self.levels != levels):
            # Regenerate stim sequence
            sequence = list(itertools.product(frequencies, levels))
            self.stim_selector = shuffled_set(sequence)
            self.frequencies = frequencies
            self.levels = levels

        if trial_type == 'go_remind':
            # Should be an easy trial. Randomly pick from selected frequencies
            # and use max level configured.
            return next(self.stim_selector)

        if trial_type == 'nogo':
            return {}

        if trial_type in ('go', 'go_forced'):
            return next(self.stim_selector)

        if trial_type in ('nogo_repeat', 'go_repeat'):
            # Repeat current stim.
            return self.current_stim

    def stim_info(self, stim):
        if stim['frequency'] is None:
            return ''
        return f' {stim["frequency"]*1e-3:.1f} Hz, {stim["level"]:.0f} dB SPL'

    def stim_waveform(self, stim):
        if stim['frequency'] is None:
            return silence(fs=self.output.fs, duration=1)
        return ramped_tone(
            **stim,
            fs=self.output.fs,
            calibration=self.output.calibration,
            duration=1,
            window='cosine-squared',
            rise_time=25e-3,
        )
