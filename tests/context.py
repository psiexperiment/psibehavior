'''
A real psiexperiment context for the tests (see `psibehavior.testing`), with
settings that reach remind, warmup and forced trials quickly.
'''
from psibehavior import testing
from psibehavior.testing import apply, change, change_expression  # noqa: F401


#: Settings that differ from the defaults the paradigm declares. The trial
#: manager's are named with its prefix.
MANAGER_SETTINGS = {
    'max_reference': 3,
    'remind_trials': 2,
    'warmup_trials': 4,
}
BEHAVIOR_SETTINGS = {
    'training_mode': False,
}


def make_context(manager_path='stubs.ToyManager', N_response=1, prefix='',
                 manager_kwargs=None, **settings):
    '''
    See `psibehavior.testing.make_context`. `settings` are applied on top of
    `MANAGER_SETTINGS` and `BEHAVIOR_SETTINGS`.
    '''
    defaults = {prefix + n: v for n, v in MANAGER_SETTINGS.items()}
    return testing.make_context(manager_path, N_response, prefix,
                                manager_kwargs,
                                **{**defaults, **BEHAVIOR_SETTINGS,
                                   **settings})
