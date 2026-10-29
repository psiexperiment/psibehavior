import sys
from pathlib import Path

import pytest

# Make the test helpers importable as top-level modules.
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def no_gui(monkeypatch):
    '''
    `deferred_call` requires a running Qt application. Run the callback
    immediately instead.
    '''
    from psi.controller import plugin as controller_plugin
    from psibehavior.paradigms import base_behavior_plugin
    for module in (controller_plugin, base_behavior_plugin):
        monkeypatch.setattr(module, 'deferred_call',
                            lambda cb, *args, **kw: cb(*args, **kw))
