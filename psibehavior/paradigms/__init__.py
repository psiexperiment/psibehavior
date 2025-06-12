from psi.experiment.api import ParadigmDescription


PATH = 'psibehavior.paradigms.'
CORE_PATH = 'psi.paradigms.core.'


COMMON_PLUGINS = [
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
        'manifest': 'cftscal.paradigms.objects.Speaker',
        'attrs': {
            'id': 'speaker_1',
            'env_prefix': 'SPEAKER_1',
        },
        'required': True,
    },
    {
        'manifest': 'cftscal.paradigms.objects.Microphone',
        'attrs': {
            'id': 'microphone_1',
            'env_prefix': 'MICROPHONE_1',
            'microphone_type': 'generic_microphone'
        },
        'required': True,
    },
]


ParadigmDescription(
    'tinnitus-2AFC', 'Tinnitus (2AFC with silence)', 'animal',
    COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_nafc.BehaviorManifest', 'attrs': {'N_response': 2, 'N_output': 1}},

        # Reward dispensers
        {
            'manifest': PATH + 'reward.PelletDispenser',
            'attrs': {'output_name': 'pellet_1', 'label': 'Pellet 1', 'event_name': 'deliver_reward_1'},
            'required': True
        },
        {
            'manifest': PATH + 'reward.PelletDispenser',
            'attrs': {'output_name': 'pellet_2', 'label': 'Pellet 2', 'event_name': 'deliver_reward_2'},
            'required': True
        },

        # Timeout
        {
            'manifest': PATH + 'timeout.Light',
            'attrs': {},
            'required': True
        },

        # Monitoring of audio
        {
            'manifest': PATH + 'monitor.SignalMonitor',
            'attrs': {
                'sources': {
                    'loopback_ttl': {'color': 'red'},
                    'loopback_1': {'color': 'black'},
                    'sync_trigger_in': {'color': 'green'},
                }
            },
            'required': True,
        },
    ],
)


ParadigmDescription(
    'hyperacusis', 'Hyperacusis (go/nogo)', 'animal',
    COMMON_PLUGINS + [
        {'manifest': PATH + 'behavior_nafc.BehaviorManifest', 'attrs': {'N_response': 1, 'N_output': 1}},
    ],
)
