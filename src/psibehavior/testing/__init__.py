'''
A real psiexperiment context for testing trial managers and the behavior
plugin without hardware or a GUI.

The parameters are the ones an experiment declares: the behavior plugin's
(`BehaviorContextItems`) and the trial manager's (`TrialManagerManifest`
generates them from the manager's `default_parameters`). Settings are
changed as the GUI changes them and only take effect when the changes are
applied.

Building a context needs the enaml application, and each context's
controller runs a dispatcher thread. In a pytest suite, create the
application once per session and call `stop_workbenches` after each test,
e.g. in conftest.py::

    @pytest.fixture(scope='session', autouse=True)
    def app():
        from enaml.application import Application
        from enaml.qt.qt_application import QtApplication
        return Application.instance() or QtApplication()

    @pytest.fixture(autouse=True)
    def stop_contexts():
        yield
        psibehavior.testing.stop_workbenches()
'''
import enaml

from psi.context.api import EnumParameter, MultiSelectParameter

with enaml.imports():
    from psi.application.workbench import PSIWorkbench
    from psi.controller.api import ControllerManifest
    from psibehavior.paradigms.trial_manager import TrialManagerManifest
    from .manifest import BehaviorContext


#: Workbenches created by `make_context`. See `stop_workbenches`.
WORKBENCHES = []


def manager_path(manager_class):
    '''
    The `manager_path` for a trial manager class.
    '''
    return f'{manager_class.__module__}.{manager_class.__qualname__}'


def make_context(manager_path, N_response=1, prefix='', manager_kwargs=None,
                 **settings):
    '''
    Return an initialized context plugin with the parameters declared by the
    behavior plugin and by the trial manager at `manager_path`.

    `settings` change the declared defaults (see `change`).
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
    change(context, **settings)
    core = workbench.get_plugin('enaml.workbench.core')
    core.invoke_command('psi.context.initialize')
    return context


def change(context, **settings):
    '''
    Change settings as the GUI does. They take effect when applied.

    Values are those the paradigm reads: e.g., `repeat_incorrect=3` selects
    the choice that has that value, and `level_list=[20, 60]` selects those
    choices of a multi-select.
    '''
    for name, value in settings.items():
        item = context.get_item(name)
        if isinstance(item, EnumParameter):
            item.expression = value
        elif isinstance(item, MultiSelectParameter):
            item.expression = repr(list(value))
            selected = sorted(item.choices[s] for s in item.selected)
            if selected != sorted(value):
                raise ValueError(f'{name}: {value!r} are not all among the '
                                 f'choices {list(item.choices.values())}')
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
