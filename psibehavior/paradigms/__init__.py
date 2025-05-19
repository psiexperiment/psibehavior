from psi.experiment.api import ParadigmDescription


PATH = 'psibehavior.paradigms.'
CORE_PATH = 'psi.paradigms.core.'


COMMON_PLUGINS = [
    {'manifest': 'psibehavior.paradigms.video.PSIVideo',
     'attrs': {
         'id': 'psivideo',
         'title': 'Video (top)',
         'port': 33331,
         'filename': 'top_recording.avi',
     }},
    {'manifest': CORE_PATH + 'signal_mixins.SignalFFTViewManifest',
     'attrs': {
         'fft_time_span': 1,
         'fft_freq_lb': 5,
         'fft_freq_ub': 24000,
         'y_label': 'Level (dB)'},
     },
]


ParadigmDescription(
    'tinnitus-2AFC', 'Tinnitus (2AFC with silence)', 'animal',
    COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_nafc.BehaviorManifest',
         'attrs': {'N_response': 2, 'N_output': 1},
         },
    ],
)


ParadigmDescription(
    'hyperacusis', 'Hyperacusis (go/nogo)', 'animal',
    COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_nafc.BehaviorManifest',
         'attrs': {'N_response': 1, 'N_output': 1},
         },
    ],
)
