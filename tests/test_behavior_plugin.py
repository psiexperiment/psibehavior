'''
State-machine tests for BehaviorPlugin using a stub context, a fake output and
manually fired timers.
'''
import threading

from atom.api import Bool, Float, Value
from enaml.workbench.api import Plugin
import pytest

from psibehavior.paradigms.behavior_nafc_plugin import (BehaviorPlugin,
                                                        NAFCTrialState as S)
from stubs import ContextNamespace, FakeOutput, ToyManager


class StubContextPlugin(Plugin):

    ns = Value()

    def get_value(self, name):
        return self.ns.get_value(name)

    def get_values(self):
        return self.ns.get_values()

    def set_value(self, name, value):
        self.ns.set_value(name, value)

    def set_values(self, values):
        self.ns.set_values(values)

    def apply_changes(self):
        self.ns.apply_changes()

    def next_setting(self, selector=None):
        self.ns.next_setting()


class HarnessPlugin(BehaviorPlugin):

    h_actions = Value(factory=list)
    h_results = Value(factory=list)
    h_timers = Value(factory=dict)
    h_ts = Float(0.0)
    h_output = Value()
    h_inline = Bool(True)
    h_threads = Value(factory=list)

    def invoke_actions(self, event_name, timestamp=None, delayed=False,
                       cancel_existing=True, kw=None, skip_errors=False,
                       wait=True):
        self.h_actions.append(event_name)
        if event_name == 'trial_end':
            self.h_results.append(kw['result'])

    def get_ts(self):
        return self.h_ts

    def get_output(self, name):
        return self.h_output

    def start_timer(self, name, duration, callback, cancel_existing=True):
        self.h_timers[name] = (duration, callback)

    def stop_timer(self, name):
        self.h_timers.pop(name, None)

    def _submit(self, fn, *args, **kw):
        if self.h_inline:
            with self.lock:
                fn(*args, **kw)
        else:
            super()._submit(fn, *args, **kw)

    def prepare_trial(self, auto_start=False):
        self.h_threads.append(threading.current_thread())
        super().prepare_trial(auto_start)

    def handle_event(self, event, timestamp=None):
        self.h_threads.append(threading.current_thread())
        super().handle_event(event, timestamp)

    # Helpers for driving the state machine.
    def fire(self, name='event'):
        duration, callback = self.h_timers.pop(name)
        self.h_ts += duration
        callback()

    def poke(self):
        self.handle_event(self.events.np_start)

    def withdraw(self):
        self.handle_event(self.events.np_end)

    def respond(self, side=1):
        self.handle_event(getattr(self.events, f'resp_{side}_start'))

    def release(self, side=1):
        self.handle_event(getattr(self.events, f'resp_{side}_end'))

    def start_by_poking(self):
        assert self.trial_state == S.waiting_for_np_start
        self.poke()
        self.fire()  # np_duration
        self.withdraw()
        assert self.trial_state == S.waiting_for_response

    @property
    def n_waveforms(self):
        return len(self.h_output.waveforms)


def make_plugin(N_response=1, **parameters):
    parameters.setdefault('remind_trials', 0)
    parameters.setdefault('warmup_trials', 0)
    plugin = HarnessPlugin(N_response=N_response, response_name='resp')
    plugin.context = StubContextPlugin(ns=ContextNamespace(**parameters))
    plugin.h_output = FakeOutput()
    plugin.trial_manager = ToyManager(plugin)
    plugin.experiment_state = 'running'
    plugin.prepare_trial()
    return plugin


def change(plugin, **parameters):
    plugin.context.ns.change(**parameters)


def test_apply_during_iti_does_not_skip_iti():
    plugin = make_plugin(target_probability=1)
    plugin.start_by_poking()
    plugin.respond(1)
    plugin.release(1)
    assert plugin.trial_state == S.waiting_for_iti
    n = plugin.n_waveforms

    change(plugin, level_list=[80])
    plugin.request_apply()
    assert plugin.context.ns.applied == 1
    assert plugin.trial_state == S.waiting_for_iti
    assert 'event' in plugin.h_timers
    assert plugin.n_waveforms == n

    plugin.fire()  # ITI
    assert plugin.trial_state == S.waiting_for_np_start
    assert plugin.context.get_value('level') == 80


def test_apply_during_timeout_does_not_cut_timeout():
    plugin = make_plugin(target_probability=0)
    plugin.start_by_poking()
    plugin.respond(1)  # False alarm
    assert 'to_start' in plugin.h_actions
    plugin.release(1)
    assert plugin.trial_state == S.waiting_for_to

    plugin.request_apply()
    assert plugin.context.ns.applied == 1
    assert plugin.trial_state == S.waiting_for_to

    plugin.fire()  # Timeout
    assert 'to_end' in plugin.h_actions
    assert plugin.trial_state == S.waiting_for_iti


@pytest.mark.parametrize('iti_duration', [0, 0.1])
def test_apply_during_trial_applied_after_result_logged(iti_duration):
    plugin = make_plugin(target_probability=1, iti_duration=iti_duration)
    plugin.start_by_poking()
    level = plugin.context.get_value('level')
    change(plugin, level_list=[80])
    plugin.request_apply()
    assert plugin._apply_latched
    assert plugin.context.ns.applied == 0
    n = plugin.n_waveforms

    plugin.respond(1)
    result = plugin.h_results[-1]
    assert result['level'] == level
    assert result['trial_type'] == 'target'
    assert result['level_list'] == [20, 40, 60]
    assert plugin.context.ns.applied == 1
    assert not plugin._apply_latched

    plugin.release(1)
    if iti_duration:
        plugin.fire()
    # Exactly one new trial, prepared with the new values.
    assert plugin.n_waveforms == n + 1
    assert plugin.context.get_value('level') == 80


def test_apply_while_waiting_reuses_prepared_trial():
    plugin = make_plugin()
    n = plugin.n_waveforms
    calls = plugin.trial_manager.next_stim_calls
    plugin.request_apply()
    assert plugin.trial_state == S.waiting_for_np_start
    assert plugin.n_waveforms == n
    assert plugin.trial_manager.next_stim_calls == calls
    assert plugin.context.get_value('trial_type') in ('target', 'reference')


def test_start_trial_button():
    plugin = make_plugin(target_probability=1)
    n = plugin.n_waveforms

    # From waiting for nose-poke: start the prepared trial.
    plugin.request_trial()
    assert plugin.trial_state == S.waiting_for_response
    assert plugin.n_waveforms == n

    # Ignored while a trial is in progress.
    plugin.request_trial()
    assert plugin.trial_state == S.waiting_for_response
    assert plugin.n_waveforms == n
    assert not plugin.h_output.waveforms[-1].cancelled

    plugin.respond(1)
    plugin.release(1)
    assert plugin.trial_state == S.waiting_for_iti

    # From the intertrial interval: skip the rest of the interval.
    plugin.request_trial()
    assert plugin.trial_state == S.waiting_for_response
    assert plugin.n_waveforms == n + 1
    assert len(plugin.h_results) == 1


def test_manual_control_prepares_once():
    plugin = make_plugin(target_probability=1, manual_control=True)
    for i in range(3):
        n = plugin.n_waveforms
        plugin.poke()  # Ignored under manual control
        assert plugin.trial_state == S.waiting_for_np_start
        plugin.withdraw()
        plugin.request_trial()
        assert plugin.n_waveforms == n
        plugin.respond(1)
        plugin.release(1)
        plugin.fire()  # ITI
    assert all(w.started for w in plugin.h_output.waveforms[:-1])


def test_poking_at_end_of_iti_does_not_start_under_manual_control():
    plugin = make_plugin(target_probability=1, manual_control=True)
    plugin.request_trial()
    plugin.respond(1)
    plugin.release(1)
    plugin.poke()
    plugin.fire()  # ITI
    assert plugin.trial_state == S.waiting_for_np_start


def test_pause_after_timeout():
    plugin = make_plugin(target_probability=0)
    plugin.start_by_poking()
    plugin.respond(1)  # False alarm
    assert plugin.trial_state == S.waiting_for_reward_end

    plugin.request_pause()
    assert plugin.experiment_state == 'running'
    assert plugin._pause_latched

    plugin.release(1)
    plugin.fire()  # Timeout
    plugin.fire()  # ITI
    assert plugin.experiment_state == 'paused'
    assert plugin.trial_state == S.waiting_for_resume
    actions = plugin.h_actions
    assert actions.index('to_end') < actions.index('experiment_paused')

    plugin.request_resume()
    assert plugin.experiment_state == 'running'
    assert plugin.trial_state == S.waiting_for_np_start


def test_resume_cancels_pending_pause():
    plugin = make_plugin(target_probability=1)
    plugin.start_by_poking()
    plugin.request_pause()
    assert plugin._pause_latched
    plugin.request_resume()
    assert not plugin._pause_latched
    plugin.respond(1)
    plugin.release(1)
    plugin.fire()  # ITI
    assert plugin.experiment_state == 'running'
    assert plugin.trial_state == S.waiting_for_np_start


def test_no_trial_starts_while_paused():
    plugin = make_plugin(target_probability=1)
    plugin.request_pause()
    assert plugin.experiment_state == 'paused'
    plugin.poke()
    assert plugin.np_active

    # Applying changes while paused must not start the prepared trial even
    # though the animal is poking.
    plugin.request_apply()
    assert plugin.trial_state == S.waiting_for_np_start

    plugin.request_resume()
    assert plugin.trial_state == S.waiting_for_response


def test_withdrawal_applies_deferred_changes():
    plugin = make_plugin(target_probability=1)
    plugin.poke()
    assert plugin.trial_state == S.waiting_for_np_duration
    first = plugin.h_output.waveforms[-1]
    change(plugin, level_list=[80])
    plugin.request_apply()
    assert plugin._apply_latched

    plugin.withdraw()
    assert plugin.context.ns.applied == 1
    assert plugin.trial_state == S.waiting_for_np_start
    assert first.cancelled
    assert plugin.context.get_value('level') == 80
    assert not plugin.h_output.waveforms[-1].started


def test_remind_requested_during_nose_poke():
    plugin = make_plugin(target_probability=0)
    plugin.poke()
    plugin.request_remind()
    assert plugin.trial_state == S.waiting_for_np_duration
    assert plugin._remind_requested

    plugin.withdraw()
    assert plugin.trial_state == S.waiting_for_np_start
    assert plugin.context.get_value('trial_subtype') == 'remind'
    assert plugin.context.get_value('trial_type') == 'target'


def test_remind_request_starts_trial_when_waiting():
    plugin = make_plugin(target_probability=0)
    plugin.request_remind()
    assert plugin.trial_state == S.waiting_for_response
    assert plugin.context.get_value('trial_subtype') == 'remind'
    assert not plugin._remind_requested


def test_reward_delivered():
    plugin = make_plugin(target_probability=1, level_list=[60])
    plugin.start_by_poking()
    plugin.respond(1)
    assert plugin.h_results[-1]['reward_delivered'] is True
    assert 'deliver_reward_1' in plugin.h_actions

    plugin.release(1)
    plugin.fire()  # ITI
    change(plugin, level_list=[20])  # Below min_reward_level
    plugin.request_apply()
    plugin.h_actions.clear()
    plugin.start_by_poking()
    plugin.respond(1)
    assert plugin.h_results[-1]['reward_delivered'] is False
    assert 'deliver_reward_1' not in plugin.h_actions


def test_reward_delivered_training_mode():
    plugin = make_plugin(target_probability=1, level_list=[60],
                         training_mode=True)
    plugin.start_by_poking()
    assert 'deliver_reward_1' in plugin.h_actions
    plugin.handle_event(plugin.events.response_duration_elapsed)
    assert plugin.h_results[-1]['reward_delivered'] is True


def test_operator_requests_run_on_dispatcher():
    '''
    Operator requests and hardware/timer events fired from several threads
    must all run on the control dispatcher and never deadlock.
    '''
    plugin = make_plugin(target_probability=0.5)
    plugin.h_threads.clear()
    plugin.h_inline = False
    dispatcher = plugin._dispatcher

    def operator():
        for i in range(50):
            plugin.request_apply()
            plugin.request_remind()
            plugin.request_trial()
            plugin.request_pause()
            plugin.request_resume()

    def animal():
        for i in range(100):
            plugin.submit_event(plugin.events.np_start)
            plugin.submit_event(plugin.events.np_end)
            plugin.submit_event(plugin.events.resp_1_start)
            plugin.submit_event(plugin.events.resp_1_end)

    threads = [threading.Thread(target=operator), threading.Thread(target=animal)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive()
        done = threading.Event()
        dispatcher.submit(done.set)
        assert done.wait(timeout=30), 'Dispatcher did not drain (deadlock?)'
        assert plugin.h_threads
        assert all(t is dispatcher._thread for t in plugin.h_threads)
    finally:
        dispatcher.stop()


def test_timer_due_at_stop_does_not_continue_trial():
    plugin = make_plugin(target_probability=1, hold_duration=0.5)
    plugin.poke()
    plugin.fire()  # np_duration
    assert plugin.trial_state == S.waiting_for_hold

    # The experiment stops while the end of the hold is already on its way
    # to the dispatcher.
    _, hold_elapsed = plugin.h_timers.pop('event')
    plugin.experiment_state = 'stopped'
    hold_elapsed()
    assert plugin.trial_state == S.waiting_for_hold
    assert 'event' not in plugin.h_timers
    assert 'trial_end' not in plugin.h_actions
