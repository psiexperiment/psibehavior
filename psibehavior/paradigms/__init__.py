from psi.experiment.api import ParadigmDescription


PATH = 'psibehavior.paradigms.'
CORE_PATH = 'psi.paradigms.core.'


COLORS = {
    # Used for TrialLog
    'go': 'lightgreen',
    'go_warmup': 'limegreen',
    'go_forced': 'limegreen',
    'go_remind': 'limegreen',
    'go_manual': 'limegreen',

    'nogo': 'lightpink',
    'nogo_forced': 'pink',
    'nogo_repeat': 'pink',
    'nogo_warmup': 'pink',
}


COMMON_PLUGINS = [
    {
        'manifest': 'psibehavior.paradigms.trial_log.TrialLogManifest',
        'required': True,
        'attrs': {'colors': COLORS,},
    },
    {
        'manifest': 'psibehavior.paradigms.video.PSIVideo',
        'attrs': {
            'id': 'psivideo',
            'title': 'Video (top)',
            'port': 33331,
            'filename': 'top_recording.avi',
        },
    },
    {
        # Make the speaker available to the behavior paradigms
        'manifest': 'cftscal.paradigms.objects.Speaker',
        'attrs': {
            'id': 'speaker_1',
            'env_prefix': 'SPEAKER_1',
        },
        'required': True,
    },
    {
        # Make the microphone available to the behavior paradigms
        'manifest': 'cftscal.paradigms.objects.Microphone',
        'attrs': {
            'id': 'microphone_1',
            'env_prefix': 'MICROPHONE_1',
            'microphone_type': 'generic_microphone'
        },
        'required': True,
    },
    {
        'manifest': 'psi.paradigms.core.signal_mixins.MultiSignalFFTViewManifest',
        'attrs': {
            'sources': {
                'microphone_1': {'color': 'black', 'apply_calibration': True},
            },
            'fft_freq_lb': 1e3,
        },
        'required': True,
    },
    {
        'manifest': 'psi.paradigms.core.signal_mixins.MultiSignalFFTViewManifest',
        'attrs': {
            'id': 'loopback_1_fft_view',
            'title': 'Loopback FFT',
            'sources': {
                'loopback_1': {'color': 'black', 'apply_calibration': True},
            },
            'fft_freq_lb': 1e2,
        },
        'required': True,
    },
    {
        'manifest': PATH + 'timeout.Light',
        'attrs': {},
        'required': True
    },
    {
        'manifest': PATH + 'logger.Logger',
        'attrs': {},
        'required': True,
    },
    {
        'manifest': PATH + 'metadata.Metadata',
        'attrs': {},
        'required': True,
    },
]


REWARD_1 = {
    'manifest': PATH + 'reward.PelletDispenser',
    'attrs': {'output_name': 'pellet_1', 'label': 'Pellet 1', 'event_name': 'deliver_reward_1'},
    'required': True
}


REWARD_2 = {
    'manifest': PATH + 'reward.PelletDispenser',
    'attrs': {'output_name': 'pellet_2', 'label': 'Pellet 2', 'event_name': 'deliver_reward_2'},
    'required': True
}


CUE = {
    'manifest': PATH + 'cue.Light',
    'attrs': {},
    'required': True
}


ParadigmDescription(
    'tinnitus-2AFC', 'Tinnitus (2AFC with silence)', 'animal',
    COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_nafc.BehaviorManifest', 'attrs': {'N_response': 2, 'N_output': 1}},
        {'manifest': PATH + 'trial_manager.TrialManagerManifest',
         'attrs': {'manager_path': 'psibehavior.paradigms.trial_manager_plugin.Tinnitus2AFCManager'},
        },
        REWARD_1,
        REWARD_2,
        CUE,
        {
            'manifest': 'psibehavior.paradigms.nafc_analysis.NAFCAnalysisManifest',
            'required': True,
            'attrs': {'colors': COLORS,},
        },
    ],
)


ParadigmDescription(
    'modulation-2AFC', 'Modulation detection (2AFC)', 'animal',
    COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_nafc.BehaviorManifest', 'attrs': {'N_response': 2, 'N_output': 1}},
        {'manifest': PATH + 'trial_manager.TrialManagerManifest',
         'attrs': {'manager_path': 'psibehavior.paradigms.trial_manager_plugin.ModulationTask'},
        },
        {'manifest': PATH + 'trial_manager.ContinuousStimManagerManifest',
         'attrs': {
             'manager_path': 'psibehavior.paradigms.trial_manager_plugin.BandlimitedFIRNoise',
             'manager_kwargs': {'output_names': ['continuous_output_1']},
         },
         'required': True,
        },
        REWARD_1,
        REWARD_2,
        {
            'manifest': 'psibehavior.paradigms.nafc_analysis.NAFCAnalysisManifest',
            'required': True,
            'attrs': {'colors': COLORS,},
        },
    ],
)


ParadigmDescription(
    'hyperacusis', 'Hyperacusis (go/nogo)', 'animal',
    COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_nafc.BehaviorManifest', 'attrs': {'N_response': 1, 'N_output': 1}},
        {'manifest': PATH + 'trial_manager.TrialManagerManifest',
         'attrs': {'manager_path': 'psibehavior.paradigms.trial_manager_plugin.HyperacusisGoNogoManager'},
        },
        REWARD_1,
        {
            'manifest': 'psibehavior.paradigms.sdt_analysis.SDTAnalysisManifest',
            'required': True,
            'attrs': {'colors': COLORS,},
        },
    ],
)
