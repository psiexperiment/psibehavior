'''
A real psiexperiment context for the tests.

The parameters are the ones an experiment declares: the behavior plugin's
(`BehaviorContextItems`) and the trial manager's (`TrialManagerManifest`
generates them from the manager's `default_parameters`). Settings are
changed as the GUI changes them, by setting each parameter's expression,
and only take effect when the changes are applied.
'''
import enaml

from psi.context.api import EnumParameter

with enaml.imports():
    from psi.application.workbench import PSIWorkbench
    from psi.controller.api import ControllerManifest
    from psibehavior.paradigms.trial_manager import TrialManagerManifest
    from behavior_context import BehaviorContext


#: Settings that differ from the defaults the paradigm declares, so that the
#: tests reach remind, warmup and forced trials quickly. The trial manager's
#: are named with its prefix.
MANAGER_SETTINGS = {
    'max_reference': 3,
    'remind_trials': 2,
    'warmup_trials': 4,
}
BEHAVIOR_SETTINGS = {
    'training_mode': False,
}

#: Workbenches created by `make_context`. See `stop_workbenches`.
WORKBENCHES = []


def make_context(manager_path='stubs.ToyManager', N_response=1, prefix='',
                 manager_kwargs=None, **settings):
    '''
    Return an initialized context plugin with the parameters declared by the
    behavior plugin and by the trial manager at `manager_path`.

    `settings` are applied on top of `MANAGER_SETTINGS` and
    `BEHAVIOR_SETTINGS` (see `change`).
    '''
    workbench = PSIWorkbench()
    manager = TrialManagerManifest(manager_path=manager_path, prefix=prefix,
                                   manager_kwargs=manager_kwargs or {})
    # BehaviorContext provides the controller's engine, so it goes first.
    workbench.register_core_plugins(None, [
        BehaviorContext(N_response=N_response),
        ControllerManifest(),
        manager,
    ])
    WORKBENCHES.append(workbench)
    context = workbench.get_plugin('psi.context')
    defaults = {prefix + n: v for n, v in MANAGER_SETTINGS.items()}
    change(context, **{**defaults, **BEHAVIOR_SETTINGS, **settings})
    core = workbench.get_plugin('enaml.workbench.core')
    core.invoke_command('psi.context.initialize')
    return context


def change(context, **settings):
    '''
    Change settings as the GUI does. They take effect when applied.

    Values are those the paradigm reads (e.g., `repeat_incorrect=3`, which
    selects the choice that has that value).
    '''
    for name, value in settings.items():
        item = context.get_item(name)
        if isinstance(item, EnumParameter):
            item.expression = value
        else:
            item.set_value(value)


def change_expression(context, name, expression):
    '''
    Change the expression of a setting as typed into the GUI (e.g., one with
    a typo, or `1/0`).
    '''
    context.get_item(name).expression = expression


def apply(context):
    '''
    Apply the changes and start the next setting, as when the operator clicks
    Apply (see `BehaviorPlugin._apply_context`).
    '''
    context.apply_changes()
    context.next_setting()


def stop_workbenches():
    '''
    Stop the controller of each workbench created so far (it runs a
    dispatcher thread).
    '''
    while WORKBENCHES:
        WORKBENCHES.pop().get_plugin('psi.controller').stop()
