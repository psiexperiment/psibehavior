import numpy as np
from psiaudio.stim import BandlimitedFIRNoiseFactory
from psiaudio.util import dbi


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
        #: Level the noise is generated at. Changes to the level (it can be
        #: changed at any time) scale the noise from there, so it plays on
        #: without restarting the filter.
        self.level = self.get_value('masker_level')

        for name, output in self.outputs.items():
            self.factories[name] = BandlimitedFIRNoiseFactory(
                fs=output.fs,
                fl=self.get_value('masker_fl')*1e3,
                fh=self.get_value('masker_fh')*1e3,
                level=self.level,
                calibration=output.calibration,
                # The band at `level` overall, as labelled (otherwise it is
                # quieter by the fraction of the spectrum it covers).
                correct_level=True,
            )

    def next(self, samples, output):
        gain = dbi(self.get_value('masker_level') - self.level)
        return self.factories[output].next(samples) * gain
