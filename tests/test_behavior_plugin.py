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
from stubs import ContextNamespace, Expression, FakeOutput, Invalid, ToyManager


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

    @property
    def expressions(self):
        return self.ns.expressions

    @property
    def symbols(self):
        return {}

    def validate_expressions(self):
        self.ns.validate_expressions()

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

    #: (kind, title) of each message shown to the operator.
    h_messages = Value(factory=list)

    #: Experiment state each time a trial started.
    h_start_states = Value(factory=list)

    #: If True, timers run on the real dispatcher (`call_later`) and every
    #: action makes a round trip through the dispatcher, as
    #: `invoke_actions(wait=True)` does. Otherwise, timers are fired by hand
    #: with `fire`.
    h_real = Bool(False)

    def invoke_actions(self, event_name, timestamp=None, delayed=False,
                       cancel_existing=True, kw=None, skip_errors=False,
                       wait=True):
        self.h_actions.append(event_name)
        if event_name == 'trial_end':
            self.h_results.append(kw['result'])
        if self.h_real:
            # Runs inline on the dispatcher thread. From any other thread
            # (e.g., the GUI holding `lock`), this waits for the dispatcher,
            # which is how the deadlock fixed in M0 arose.
            self._dispatcher.submit_sync(lambda: None)

    def get_ts(self):
        return self.h_ts

    def get_output(self, name):
        return self.h_output

    def start_timer(self, name, duration, callback, cancel_existing=True):
        if self.h_real:
            super().start_timer(name, duration, callback, cancel_existing)
        else:
            self.h_timers[name] = (duration, callback)

    def stop_timer(self, name):
        if self.h_real:
            super().stop_timer(name)
        else:
            self.h_timers.pop(name, None)

    def start_trial(self):
        self.h_start_states.append(self.experiment_state)
        super().start_trial()

    def _show_message(self, kind, title, message):
        self.h_messages.append((kind, title))

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


def drain(dispatcher, timeout=30):
    done = threading.Event()
    dispatcher.submit(done.set)
    assert done.wait(timeout=timeout), 'Dispatcher did not drain (deadlock?)'


@pytest.mark.parametrize('real', [False, True])
def test_operator_requests_run_on_dispatcher(real):
    '''
    Operator requests and hardware/timer events fired from several threads
    must all run on the control dispatcher and never deadlock.

    With `real`, timers run on the dispatcher's own `call_later` and every
    action makes a round trip through the dispatcher, so a request handled
    on the calling thread while holding `lock` (as before M0) deadlocks and
    fails this test.
    '''
    durations = dict(np_duration=0.002, iti_duration=0.002, to_duration=0.002,
                     response_duration=0.005)
    plugin = make_plugin(target_probability=0.5, **(durations if real else {}))
    plugin.h_threads.clear()
    plugin.h_inline = False
    plugin.h_real = real
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

    def stop():
        plugin.experiment_state = 'stopped'
        plugin.stop_all_timers()

    threads = [threading.Thread(target=operator), threading.Thread(target=animal)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), 'Request thread blocked (deadlock?)'
        drain(dispatcher)
        assert plugin.h_threads
        assert all(t is dispatcher._thread for t in plugin.h_threads)

        if real:
            # Stop on the dispatcher so that no timer fires afterwards.
            dispatcher.submit(stop)
            drain(dispatcher)
            waveforms = plugin.h_output.waveforms
            started = [w for w in waveforms if w.started]
            queued = [w for w in waveforms if not w.started and not w.cancelled]
            in_flight = plugin.trial_state in (S.waiting_for_hold,
                                               S.waiting_for_response)
            assert started
            # No trial ever started while paused.
            assert 'paused' not in plugin.h_start_states
            # At most one prepared trial is waiting on the output.
            assert len(queued) <= 1
            # Every trial that started was logged exactly once (except one
            # still in progress when the experiment stopped).
            assert len(started) == len(plugin.h_results) + in_flight
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


def test_invalid_apply_rejected_when_requested():
    plugin = make_plugin(target_probability=1)
    plugin.start_by_poking()
    change(plugin, level_list=Invalid())
    plugin.request_apply()
    # Reported now, not latched for the end of the trial.
    assert plugin.h_messages == [('critical', 'Cannot apply changes')]
    assert not plugin._apply_latched

    plugin.respond(1)
    plugin.release(1)
    assert plugin.trial_state == S.waiting_for_iti
    assert plugin.context.ns.applied == 0


@pytest.mark.parametrize('iti_duration', [0, 0.1])
def test_latched_apply_that_fails_does_not_hang_trial(iti_duration):
    '''
    The changes are valid when Apply is clicked, but are edited (e.g., a typo)
    before the trial ends. The trial must still end normally.
    '''
    plugin = make_plugin(target_probability=1, iti_duration=iti_duration)
    plugin.start_by_poking()
    change(plugin, level_list=[80])
    plugin.request_apply()
    assert plugin._apply_latched
    change(plugin, target_probability=Invalid())

    plugin.respond(1)
    assert len(plugin.h_results) == 1
    assert plugin.h_messages == [('critical', 'Cannot apply changes')]
    # Cleared so that Apply can be clicked again.
    assert not plugin._apply_latched
    assert not plugin._apply_requested

    plugin.release(1)
    if iti_duration:
        plugin.fire()  # ITI
    assert plugin.trial_state == S.waiting_for_np_start
    # The next trial runs with the settings that were in effect.
    assert plugin.context.get_value('level') in (20, 40, 60)
    plugin.start_by_poking()
    plugin.respond(1)
    assert len(plugin.h_results) == 2

    # Fixing the typo and applying again works.
    change(plugin, target_probability=1)
    plugin.release(1)
    if iti_duration:
        plugin.fire()  # ITI
    plugin.request_apply()
    assert plugin.context.ns.applied == 1
    assert plugin.context.get_value('level') == 80


def test_latched_apply_that_fails_at_withdrawal():
    plugin = make_plugin(target_probability=1)
    plugin.poke()
    first = plugin.h_output.waveforms[-1]
    change(plugin, level_list=[80])
    plugin.request_apply()
    change(plugin, target_probability=Invalid())

    plugin.withdraw()
    assert plugin.trial_state == S.waiting_for_np_start
    assert plugin.h_messages == [('critical', 'Cannot apply changes')]
    # The prepared trial is kept.
    assert not first.cancelled
    plugin.start_by_poking()
    assert first.started


def test_apply_that_cannot_be_evaluated_rejected_when_requested():
    plugin = make_plugin(target_probability=1)
    change(plugin, iti_duration=Expression('1/0'))
    plugin.request_apply()
    assert plugin.h_messages == [('critical', 'Cannot apply changes')]
    assert plugin.context.ns.applied == 0
    assert plugin.context.get_value('iti_duration') == 0.1


@pytest.mark.parametrize('name', ['iti_duration', 'to_duration'])
def test_latched_apply_that_cannot_be_evaluated_keeps_settings(name):
    '''
    The changes pass validation but fail when evaluated (e.g., `1/0`). They
    would only fail after the trial ends, when the timeout or intertrial
    interval starts, which would leave no timer running.
    '''
    plugin = make_plugin(target_probability=0)
    plugin.start_by_poking()
    change(plugin, level_list=[80])
    plugin.request_apply()
    assert plugin._apply_latched
    change(plugin, **{name: Expression('1/0')})

    plugin.respond(1)  # False alarm, so there is a timeout
    assert len(plugin.h_results) == 1
    assert plugin.h_messages == [('critical', 'Cannot apply changes')]
    assert not plugin._apply_latched
    assert plugin.context.ns.applied == 0

    plugin.release(1)
    plugin.fire()  # Timeout
    plugin.fire()  # ITI
    assert plugin.trial_state == S.waiting_for_np_start
    plugin.start_by_poking()
    plugin.respond(1)
    assert len(plugin.h_results) == 2


def test_state_parameter_using_trial_value_is_applied():
    # `level` is only known once a trial is prepared, so it cannot be
    # evaluated when Apply is clicked.
    plugin = make_plugin(target_probability=1)
    change(plugin, iti_duration=Expression('level / 100'))
    plugin.request_apply()
    assert plugin.h_messages == []
    assert plugin.context.ns.applied == 1


@pytest.mark.parametrize('request_name', ['request_trial', 'request_remind'])
def test_ending_iti_early_takes_latched_pause(request_name):
    plugin = make_plugin(target_probability=1, remind_trials=0,
                         warmup_trials=0)
    plugin.start_by_poking()
    plugin.request_pause()
    assert plugin._pause_latched
    plugin.respond(1)
    plugin.release(1)
    assert plugin.trial_state == S.waiting_for_iti
    n_started = len(plugin.h_start_states)

    getattr(plugin, request_name)()
    assert plugin.experiment_state == 'paused'
    assert plugin.trial_state == S.waiting_for_resume
    assert len(plugin.h_start_states) == n_started
    assert 'experiment_paused' in plugin.h_actions

    plugin.request_resume()
    assert plugin.trial_state == S.waiting_for_np_start
    if request_name == 'request_remind':
        # The remind trial is the first trial after resuming.
        assert plugin._remind_requested
        plugin.start_by_poking()
        assert plugin.trial_manager._played.source == 'user_remind'


def fail_next_prepare(plugin):
    '''
    Apply a change that the trial manager cannot plan with.
    '''
    change(plugin, level_list=[-1])
    plugin.request_apply()
    assert plugin.trial_state == S.waiting_for_trial_start
    assert plugin.h_messages[-1] == ('critical', 'Unable to prepare the next trial')
    assert plugin.trial_state_str.startswith('Unable to prepare')


def test_failed_prepare_recovers_on_apply():
    plugin = make_plugin(target_probability=1)
    n = plugin.n_waveforms
    fail_next_prepare(plugin)

    # The animal cannot start a trial.
    plugin.poke()
    assert plugin.trial_state == S.waiting_for_trial_start
    plugin.withdraw()
    assert plugin.n_waveforms == n

    change(plugin, level_list=[80])
    plugin.request_apply()
    assert plugin.trial_state == S.waiting_for_np_start
    assert plugin.context.get_value('level') == 80
    plugin.start_by_poking()
    assert plugin.h_output.waveforms[-1].started


def test_failed_prepare_reverted_plays_original_trial():
    '''
    Reverting the change reuses the trial prepared before it. That trial
    must not have been cancelled (it would play silently).
    '''
    plugin = make_plugin(target_probability=1)
    first = plugin.h_output.waveforms[-1]
    level = plugin.context.get_value('level')
    fail_next_prepare(plugin)
    assert not first.cancelled

    change(plugin, level_list=[20, 40, 60])
    plugin.request_apply()
    assert plugin.trial_state == S.waiting_for_np_start
    assert plugin.context.get_value('level') == level
    plugin.start_by_poking()
    assert first.started and not first.cancelled
    plugin.respond(1)
    assert plugin.h_results[-1]['stim_complete']


def test_failed_prepare_at_end_of_iti():
    plugin = make_plugin(target_probability=1)
    plugin.start_by_poking()
    change(plugin, level_list=[-1])
    plugin.request_apply()
    plugin.respond(1)
    plugin.release(1)
    plugin.fire()  # ITI
    assert plugin.trial_state == S.waiting_for_trial_start
    assert plugin.h_messages[-1] == ('critical', 'Unable to prepare the next trial')


@pytest.mark.parametrize('request_name', ['request_trial', 'request_remind'])
def test_failed_prepare_recovers_on_request(request_name):
    plugin = make_plugin(target_probability=1)
    fail_next_prepare(plugin)
    # The change was applied, so the settings must be fixed first; until
    # then, each request fails the same way.
    getattr(plugin, request_name)()
    assert plugin.trial_state == S.waiting_for_trial_start
    plugin.context.ns.parameters['level_list'] = [60]
    getattr(plugin, request_name)()
    assert plugin.trial_state == S.waiting_for_response


def test_failed_prepare_recovers_on_resume():
    plugin = make_plugin(target_probability=1)
    fail_next_prepare(plugin)
    plugin.request_pause()
    assert plugin.experiment_state == 'paused'
    change(plugin, level_list=[60])
    plugin.request_apply()
    plugin.request_resume()
    assert plugin.trial_state == S.waiting_for_np_start
    assert plugin.context.get_value('level') == 60


@pytest.mark.parametrize('request_name', ['request_apply', 'request_pause',
                                          'request_resume', 'request_remind',
                                          'request_trial',
                                          'start_random_behavior'])
def test_requests_ignored_after_stop(request_name):
    plugin = make_plugin(target_probability=1)
    change(plugin, level_list=[80])
    n = plugin.n_waveforms
    plugin.experiment_state = 'stopped'
    getattr(plugin, request_name)()
    assert plugin.experiment_state == 'stopped'
    assert plugin.n_waveforms == n
    assert plugin.context.ns.applied == 0
    assert not plugin.random_behavior_mode
    assert 'random_behavior' not in plugin.h_timers


def test_latched_pause_after_stop_does_not_pause():
    plugin = make_plugin(target_probability=1)
    plugin.poke()
    plugin.request_pause()
    assert plugin._pause_latched
    plugin.experiment_state = 'stopped'
    plugin._pause_now()
    assert plugin.experiment_state == 'stopped'
    assert not plugin._pause_latched
    assert 'experiment_paused' not in plugin.h_actions


def test_random_behavior_toggle_stays_in_order():
    '''
    A quick on-off toggle must leave random behavior off, even when the
    dispatcher is busy when the toggle happens.
    '''
    plugin = make_plugin()
    plugin.h_inline = False
    dispatcher = plugin._dispatcher
    release = threading.Event()
    try:
        # Keep the dispatcher busy so both requests are queued.
        dispatcher.submit(release.wait, 10)
        plugin.start_random_behavior()
        plugin.stop_random_behavior()
        release.set()
        drain(dispatcher)
        assert not plugin.random_behavior_mode
        assert 'random_behavior' not in plugin.h_timers
    finally:
        release.set()
        dispatcher.stop()
