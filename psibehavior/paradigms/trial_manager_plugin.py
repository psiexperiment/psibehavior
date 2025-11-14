import logging
log = logging.getLogger(__name__)

import itertools

from atom.api import Atom, Int, Str
import numpy as np

from psi.context.api import counterbalanced, shuffled_set
from psiaudio.stim import bandlimited_noise, bandlimited_fir_noise, apply_cos2envelope, apply_sam_envelope

from psiaudio.stim import silence, stm, ramped_tone, BandlimitedFIRNoiseFactory
from psiaudio.util import octave_band_freqs, octave_space


class BaseContinuousStimManager:

    #: List of parameters to add to GUI. Eliminates the need for a custom
    #: `TrialManagerManifest` class if you only want to add new parameters to
    #: the GUI.
    default_parameters = []

    #: Default group name for each parameter in `default_parameters`. If None,
    #: `group_name`, must be provided for each parameter.
    default_group_name = None

    def get_value(self, value_name, *args, **kw):
        '''
        Get value of parameter, applying prefix if needed
        '''
        return self.context.get_value(f'{self.prefix}{value_name}', *args, **kw)

    def __init__(self, controller, output_names, prefix=''):
        self.controller = controller
        self.context = controller.context
        self.prefix = prefix

        # Link the output to the callback.
        outputs = {}
        for output_name in output_names:
            output = controller.get_output(output_name)
            output.callback = self.next
            output.active = True
            outputs[output_name] = output
        self.outputs = outputs

    def initialize(self):
        raise NotImplementedError

    def next(self, samples, channel):
        # If channel is not None, return a N dimensional array since the code
        # wants to set all channels at once.
        raise NotImplementedError


class Silence(BaseContinuousStimManager):

    def __init__(self, controller, output_names, prefix=''):
        super().__init__(controller, output_names, prefix)

    def next(self, samples, output):
        return np.zeros(samples)


class BandlimitedFIRNoise(BaseContinuousStimManager):

    default_parameters = [
        {
            'name': 'masker_fl',
            'label': 'Lower Freq. (kHz)',
            'default': 0.5,
            'scope': 'experiment',
        },
        {
            'name': 'masker_fh',
            'label': 'Upper Freq. (kHz)',
            'default': 16,
            'scope': 'experiment',
        },
        {
            'name': 'masker_level',
            'label': 'Level (dB SPL)',
            'scope': 'arbitrary',
            'default': 20,
        },
    ]

    default_group_name = 'Bandlimited FIR noise masker'

    def __init__(self, controller, output_names, prefix=''):
        super().__init__(controller, output_names, prefix)
        self.factories = {}
        self.level = self.get_value('masker_level')

        for name, output in self.outputs.items():
            self.factories[name] = BandlimitedFIRNoiseFactory(
                fs=output.fs,
                fl=self.get_value('masker_fl')*1e3,
                fh=self.get_value('masker_fh')*1e3,
                level=self.level,
                calibration=output.calibration,
            )


    def next(self, samples, output):
        if self.level != (level := self.get_value('masker_level')):
            self.level = level
            self.factories[output].update_level(self.level)
        return self.factories[output].next(samples)


class BaseTrialManager:

    #: List of parameters to add to GUI. Eliminates the need for a custom
    #: `TrialManagerManifest` class if you only want to add new parameters to
    #: the GUI.
    default_parameters = []

    #: Default group name for each parameter in `default_parameters`. If None,
    #: `group_name`, must be provided for each parameter.
    default_group_name = None

    def __init__(self, controller, prefix=''):
        self.controller = controller
        self.context = controller.context
        self.prefix = prefix
        self.trial_number = 0

    def get_value(self, value_name, *args, **kw):
        '''
        Get value of parameter, applying prefix if needed
        '''
        return self.context.get_value(f'{self.prefix}{value_name}', *args, **kw)

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
            'name': 'nbn_n',
            'label': 'NBN per set',
            'default': 5,
            'scope': 'arbitrary',
        },
        {
            'name': 'sam_n',
            'label': 'SAM per set',
            'default': 3,
            'scope': 'arbitrary',
        },
        {
            'name': 'silence_n',
            'label': 'Silence per set',
            'default': 2,
            'scope': 'arbitrary',
        },
        {
            'name': 'reward_punish_silent',
            'label': 'Reward/Punish silent trials?',
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
        {
            'name': 'trial_rewarded',
            'label': 'Trial rewarded?',
            'type': 'Result',
        },
    ]

    def __init__(self, controller, prefix=''):
        super().__init__(controller, prefix)
        self.output = self.controller.get_output('output_1')
        self.sync_trigger = self.controller.get_output('sync_trigger_out')

        # Attributes that need to be configured by `prepare_trial`.
        self.trial_state_str = ''
        self.prior_response = None
        self.stim_n = 0
        self.freq = None
        self.reward_rng = np.random.default_rng()

        self.stim_config = {
            'NBN': {
                'side': 1,
            },
            'SAM': {
                'side': 2,
            },
            'silence': {
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
        freq = self.get_value('frequencies')
        nbn_n = self.get_value('nbn_n')
        sam_n = self.get_value('sam_n')
        silence_n = self.get_value('silence_n')
        n = nbn_n + sam_n + silence_n
        if n == 0:
            raise ValueError('Must have at least one stimulus configured')

        if self.stim_n != (nbn_n, sam_n, silence_n):
            # Stim sequence has not been initialized or list of stimuli to test
            # has changed. Counterbalance across two "sets".
            stim_seq = ['NBN'] * nbn_n + ['SAM'] * sam_n + ['silence'] * silence_n
            self.stim_selector = counterbalanced(stim_seq, n=n*2)
            self.stim_n = (nbn_n, sam_n, silence_n)

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
        repeat_mode = self.get_value('repeat_incorrect')
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

        reward_rate = self.get_value('reward_rate')
        reward_silent = self.get_value('reward_punish_silent')
        side = self.stim_config[self.current_trial]['side']
        if (self.current_trial == 'silence') and not reward_silent:
            # Don't reward on a silent trial or punish
            response_condition = [side]
            reward_condition = []
            timeout_condition = []
            rewarded = False
        elif self.reward_rng.uniform() < reward_rate:
            # Reward the trial
            response_condition = [side]
            reward_condition = [side]
            timeout_condition = [1, 2]
            timeout_condition.remove(side)
            rewarded = True
        else:
            # Don't reward the trial
            response_condition = [side]
            reward_condition = []
            timeout_condition = [1, 2]
            timeout_condition.remove(side)
            rewarded = False

        # Prepare a string representation for GUI
        stim_info = self.current_trial
        if self.current_trial == 'NBN':
            stim_info = f'{stim_info} {self.current_freq*1e-3:.1f} kHz'
        if self.current_trial in ('NBN', 'SAM'):
            stim_info = f'{stim_info} seed={self.current_seed}'
        self.controller.trial_state_str = \
            f'Trial {self.trial_number + 1}, ' \
            f'{"repeat " if trial_subtype is not None else ""}{stim_info}, ' \
            f'respond on {response_condition} ' \
            f'{"" if rewarded else "(no reward)"}'

        self.context.set_value('trial_rewarded', rewarded)
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

        return response_condition, reward_condition, timeout_condition

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

    def __init__(self, controller, prefix=''):
        super().__init__(controller, prefix)
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
        n_remind = self.get_value('remind_trials')
        n_warmup = self.get_value('warmup_trials')
        go_probability = self.get_value('go_probability')
        repeat_mode = self.get_value('repeat_incorrect')
        max_nogo = self.get_value('max_nogo')

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

        if self.trial_type.startswith('nogo'):
            return [0], [0], [1]
        else:
            return [1], [1], [0]

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
            'label': 'Frequency (Hz)',
            'compact_label': 'Freq (Hz)',
            'type': 'Result',
        },
        {
            'name': 'level',
            'label': 'Level (dB SPL)',
            'type': 'Result',
        },
    ] + GoNogoTrialManager.default_parameters

    def __init__(self, controller, prefix=''):
        super().__init__(controller, prefix)
        self.output = self.controller.get_output('output_1')
        self.sync_trigger = self.controller.get_output('sync_trigger_out')
        # Attributes that need to be configured by `prepare_trial`.
        self.frequencies = None
        self.levels = None

    def next_stim(self, trial_type, trial_subtype):
        # First, check to see if any context values have changed and, if so,
        # update the selector.
        frequencies = self.get_value('frequency_list')
        levels = self.get_value('level_list')

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


class NAFCTrialManager(BaseTrialManager):

    def __init__(self, controller, prefix=''):
        super().__init__(controller, prefix)
        self.prior_response = None
        self.prior_score = None
        self.rng = np.random.default_rng()

    def prepare_trial(self):
        repeat_mode = self.get_value('repeat_incorrect')
        if self.prior_response is None:
            trial_subtype = None
            self.current_stim = self.next_stim()
        elif repeat_mode == 1 and self.prior_response == 'early_np':
            trial_subtype = 'repeat'
        elif repeat_mode == 2 and self.prior_score != self.controller.scores.correct:
            trial_subtype = 'repeat'
        else:
            trial_subtype = None
            self.current_stim = self.next_stim()

        self.context.set_value('trial_subtype', trial_subtype)
        self.context.set_values(self.current_stim)
        self.controller.trial_state_str = self.get_trial_state_str(self.current_stim)
        waveform = self.stim_waveform(self.current_stim)
        with self.output.engine.lock:
            self.output.set_waveform(waveform)
        return self.get_conditions(self.current_stim)

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


class ModulationTask(NAFCTrialManager):

    default_group_name = 'Stimuli'

    default_parameters = [
        {
            'name': 'fc_list',
            'label': 'Noise center frequencies (kHz)',
            'default': ['8'],
            'scope': 'arbitrary',
            'type': 'MultiSelectParameter',
            'choices': {str(f): f for f in [1, 2, 4, 8, 16, 32]},
            'quote_values': False,
            'button_width': 40,
            'n_cols': 6,
        },
        {
            'name': 'stm_depth_list',
            'label': 'STM depths (dB)',
            'default': ['24'],
            'scope': 'arbitrary',
            'type': 'MultiSelectParameter',
            'choices': {str(d): d for d in [3, 6, 9, 12, 15, 18, 21, 24]},
            'quote_values': False,
            'button_width': 40,
            'n_cols': 4,
        },
        {
            'name': 'target_probability',
            'label': 'Target prob. (frac.)',
            'default': 0.5,
            'scope': 'arbitrary',
        },
        {
            'name': 'bw',
            'label': 'Bandwidth (octaves)',
            'default': 1,
            'scope': 'arbitrary',
        },
        {
            'name': 'cpo',
            'label': 'Ripple (cycles per octave)',
            'default': 2,
            'scope': 'arbitrary',
        },
        {
            'name': 'cps',
            'label': 'Temporal rate (Hz)',
            'default': 4,
            'scope': 'arbitrary',
        },
        {
            'name': 'center_level',
            'label': 'Center level (dB SPL)',
            'default': 60,
            'scope': 'arbitrary',
        },
        {
            'name': 'level_range',
            'label': 'Level range (dB)',
            'default': 10,
            'scope': 'arbitrary',
        },
        {
            'name': 'rise_time',
            'label': 'Envelope rise time (sec)',
            'default': 25e-3,
            'scope': 'arbitrary',
        },
        {
            'name': 'stm_depth',
            'label': 'STM depth',
            'type': 'Result',
        },
        {
            'name': 'fc',
            'label': 'Fc (kHz)',
            'type': 'Result',
        },
        {
            'name': 'actual_level',
            'label': 'Level (dB SPL)',
            'type': 'Result',
        },
    ]

    def __init__(self, controller, prefix=''):
        super().__init__(controller, prefix)
        self.output = self.controller.get_output('output_1')
        self.sync_trigger = self.controller.get_output('sync_trigger_out')
        self.stm_depth_list = []
        self.fc_list = []

    def next_stim(self):
        # Check if any of the roving values have changed and update the
        # selectors as needed 
        stm_depth_list = self.get_value('stm_depth_list')
        fc_list = self.get_value('fc_list')
        if self.stm_depth_list != stm_depth_list:
            self.stm_depth_selector = counterbalanced(stm_depth_list, len(stm_depth_list) * 2)
            self.stm_depth_list = stm_depth_list
        if self.fc_list != fc_list:
            self.fc_selector = counterbalanced(fc_list, len(fc_list) * 4)
            self.fc_list = fc_list
        target_probability = self.get_value('target_probability')

        # Calculate the next depth and center frequency
        stm_depth = 0 if self.rng.uniform() >= target_probability else next(self.stm_depth_selector)
        fc = next(self.fc_selector)
        trial_type = 'reference' if stm_depth == 0 else 'target'
        level_range = self.get_value('level_range')
        level = self.get_value('center_level')
        actual_level = int(level + self.rng.uniform(-level_range / 2, level_range / 2))
        return {
            'stm_depth': stm_depth,
            'fc': fc,
            'trial_type': trial_type,
            'actual_level': actual_level,
        }

    def get_trial_state_str(self, stim):
        return f'{stim["trial_type"].capitalize()}: ' \
            f'depth {stim["stm_depth"]} dB, Fc {stim["fc"]} Hz, level {stim["actual_level"]} dB SPL'

    def get_conditions(self, stim):
        side = 1 if stim['stm_depth'] == 0 else 2
        response_condition = [side]
        reward_condition = [side]
        timeout_condition = [1, 2]
        timeout_condition.remove(side)
        return response_condition, reward_condition, timeout_condition

    def stim_waveform(self, stim):
        frequency = {
            'fc': stim['fc'] * 1e3,
            'octaves': self.get_value('bw'),
            'rolloff_octaves': 0.25,
            'rolloff': 16,
        }
        waveform = stm(
            frequency=frequency,
            depth=stim['stm_depth'],
            cpo=self.get_value('cpo'),
            cps=self.get_value('cps'),
            fs=self.output.fs,
            duration=1,
            mod_type='exp',
            calibration=self.output.calibration,
            level=stim['actual_level'],
        )
        waveform = apply_cos2envelope(
            waveform,
            fs=self.output.fs,
            rise_time=self.get_value('rise_time'),
        )
        return waveform
