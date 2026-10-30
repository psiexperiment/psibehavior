import logging
log = logging.getLogger(__name__)
import enum
from functools import partial

from threading import RLock

from atom.api import Bool, Int, Typed, Value
import enaml
from enaml.core.api import Conditional
from enaml.application import deferred_call
from enaml.workbench.api import Extension
from enaml.workbench.core.api import Command


from psi.context.expression import ExpressionNamespace
from psi.controller.api import (ControllerManifest, ControllerPlugin,
                                ExperimentEvent,
                                ExperimentState)

from psi.util import rpc


class TrialState(enum.Enum):
    pass


################################################################################
# Base controllers
################################################################################
class BaseBehaviorPlugin(ControllerPlugin):

    #: Must define event_map as an attribute on the subclass. Keys will be a
    #: tuple of (edge, event_type) where edge is either 'rising' or 'falling'
    #: and event_type indicates the type of event (e.g., reward contact). The
    #: key ('rising', 'reward_contact') indicates that a rising edge was
    #: detected on the 'reward_contact' input, indicating that the animal has
    #: begun attempting ot retrive a reward. See `handle_event`.
    event_map = {}

    #: Has a remind trial been requested?
    _remind_requested = Bool(False)

    #: Info about the trial that will get saved to the trial log
    trial_info = Typed(dict, ())

    #: Current state of the trial (see BaseGoNogoTrialState for some possible
    #: states).
    trial_state = Typed(TrialState)

    #: Lock object to ensure concurrency.
    lock = Value()

    #: Apply and pause requests that could not take effect immediately. These
    #: are owned by the control dispatcher. The GUI binds to the
    #: `_apply_requested` and `_pause_requested` flags on `ControllerPlugin`,
    #: which mirror these via `deferred_call`.
    _apply_latched = Bool(False)
    _pause_latched = Bool(False)

    #: Names of the parameters the state machine reads as it moves between
    #: states (e.g., durations). These are evaluated before changes are
    #: applied: an expression that is valid but cannot be evaluated (e.g.,
    #: `1/0`) would otherwise fail partway through a state transition and
    #: leave the state machine stuck.
    state_parameters = ()

    def _default_lock(self):
        return RLock()

    def can_modify(self):
        return True

    ############################################################################
    # Operator requests
    #
    # Toolbar buttons (Apply, Pause, Request Remind, Start Trial, simulated
    # events) are clicked on the GUI thread. Timer and hardware events are
    # handled on the control dispatcher thread. To avoid races (and deadlocks
    # from holding `lock` while waiting on the dispatcher), every request is
    # handed to the dispatcher and the GUI never waits for it.
    ############################################################################
    def _submit(self, fn, *args, **kw):
        '''
        Run `fn` on the control dispatcher thread without waiting.
        '''
        def run():
            with self.lock:
                fn(*args, **kw)
        self._dispatcher.submit(run)

    def _submit_request(self, fn, *args, **kw):
        '''
        Run the operator request `fn` on the control dispatcher thread
        without waiting.

        The request is dropped if, by the time it runs, the experiment is
        not running or paused (e.g., a click that was queued behind Stop).
        '''
        def run():
            if self.experiment_state not in ('running', 'paused'):
                log.info('Ignoring %s since the experiment is %s',
                         fn.__name__, self.experiment_state)
                return
            fn(*args, **kw)
        self._submit(run)

    def submit_event(self, event):
        '''
        Handle `event` (e.g., simulated from the toolbar) on the dispatcher.
        '''
        self._submit(self.handle_event, event)

    def request_apply(self):
        self._submit_request(self._request_apply)

    def request_pause(self):
        self._submit_request(self._request_pause)

    def request_resume(self):
        self._submit_request(self._request_resume)

    def request_remind(self):
        self._submit_request(self._request_remind)

    def report_error(self, title, message):
        '''
        Tell the operator about a problem found on the dispatcher thread.

        Exceptions raised on the dispatcher are only logged, so anything the
        operator needs to act on must be reported here.
        '''
        log.error('%s: %s', title, message)
        deferred_call(self._show_message, 'critical', title, message)

    def _show_message(self, kind, title, message):
        # Runs on the GUI thread. `kind` names a function in
        # enaml.stdlib.message_box (e.g., 'critical' or 'warning').
        with enaml.imports():
            from enaml.stdlib import message_box
        getattr(message_box, kind)(None, title, message)

    def check_changes(self):
        '''
        Raise ValueError if the changes that have not been applied yet
        cannot be applied.

        Besides the checks made by the context (syntax and unknown names),
        this evaluates the pending expressions for `state_parameters`.
        '''
        self.context.validate_expressions()
        expressions = self.context.expressions
        namespace = ExpressionNamespace(expressions, dict(self.context.symbols))
        for name in self.state_parameters:
            if name not in expressions:
                # Roved, so the value is only known once a trial is prepared.
                continue
            try:
                namespace.get_value(name)
            except NameError:
                # The names were checked above, so this depends on a value
                # that is only known once a trial is prepared.
                continue
            except Exception as e:
                raise ValueError(
                    f'"{name}": expression {str(expressions[name])!r} cannot '
                    f'be evaluated ({type(e).__name__}: {e}).') from e

    def _request_apply(self):
        # Check the changes now so that a typo is reported when Apply is
        # clicked rather than when the latched request is acted on.
        try:
            self.check_changes()
        except Exception as e:
            self.report_error('Cannot apply changes', str(e))
            return
        if not self.apply_changes():
            log.info('Apply requested')
            self._set_apply_latch(True)

    def _request_pause(self):
        if not self.pause_experiment():
            log.info('Pause requested')
            self._set_pause_latch(True)

    def _request_resume(self):
        self.resume_experiment()

    def _request_remind(self):
        self._remind_requested = True
        if self.can_modify():
            self.prepare_trial(auto_start=True)

    def _set_apply_latch(self, value):
        self._apply_latched = value
        deferred_call(setattr, self, '_apply_requested', value)

    def _set_pause_latch(self, value):
        self._pause_latched = value
        deferred_call(setattr, self, '_pause_requested', value)

    ############################################################################
    # ControllerPlugin template methods (run on the dispatcher)
    ############################################################################
    def pause_experiment(self):
        if self.can_modify():
            self._pause_now()
            return True
        return False

    def _pause_now(self):
        if self.experiment_state != 'running':
            # E.g., stopped. Pausing would revive the experiment state.
            self._set_pause_latch(False)
            return
        self.experiment_state = 'paused'
        self._set_pause_latch(False)
        self.invoke_actions('experiment_paused', self.get_ts())

    def resume_experiment(self):
        if self.experiment_state != 'paused':
            # The pause has not taken effect yet. Cancel it.
            if self._pause_latched:
                log.info('Pending pause cancelled')
                self._set_pause_latch(False)
            return True
        self.experiment_state = 'running'
        self.invoke_actions('experiment_resumed', self.get_ts())
        self.prepare_trial()
        return True

    def apply_changes(self):
        if self.can_modify():
            if self._apply_context():
                self.prepare_trial()
            return True
        return False

    def _apply_context(self):
        '''
        Apply pending changes to the context and clear the apply request.

        This clears any values set in the context (e.g., the stimulus of the
        current trial), so it must not be called between preparing a trial and
        logging its result.

        Returns True if the changes were applied. If they could not be applied
        (e.g., an invalid expression), the operator is told, the request is
        cleared so that Apply can be clicked again once the problem is fixed,
        and False is returned. This never raises: callers apply changes at
        safe points in the state machine, and the state machine must carry on
        whether or not the changes were applied.
        '''
        log.info('Applying changes')
        self._set_apply_latch(False)
        try:
            # The changes may have been edited since Apply was clicked, so
            # check them again before they replace the current settings.
            self.check_changes()
            self.context.apply_changes()
        except Exception as e:
            log.exception('Unable to apply changes')
            self.report_error('Cannot apply changes', str(e))
            return False
        self.context.next_setting()
        return True

    def et_callback(self, name, edge, event_time):
        event = self.event_map[edge, name]
        self.handle_event(event, event_time)

    def handle_event(self, event, timestamp=None):
        '''
        Given the current experiment state, process the appropriate response
        for the event that occured. Depending on the experiment state, a
        particular event may not be processed.
        '''
        with self.lock:
            # Only events generated by engine callbacks will have a timestamp.
            # Since we want all timing information to be in units of the analog
            # output sample clock, we will capture the value of the sample clock if
            # a timestamp is not provided. Since there will be some delay between
            # the time the event occurs and the time we read the analog clock, the
            # timestamp won't be super-accurate. However, it's not super-important
            # since these events are not reference points around which we would do
            # a perievent analysis. Important reference points would include
            # nose-poke initiation and withdraw, reward contact, sound onset,
            # lights on, lights off. These reference points will be tracked via
            # NI-DAQmx or can be calculated (i.e., we know exactly when the target
            # onset occurs because we precisely specify the location of the target
            # in the analog output buffer).
            if self.experiment_state == 'stopped':
                # Stopping cancels the timers, but one that was already due
                # (e.g., the end of the hold) may still be waiting to run.
                # It must not carry the trial on: that would start new
                # timers and log a trial after the data have been saved.
                log.info('Ignoring %s since the experiment has stopped', event)
                return

            if timestamp is None:
                timestamp = self.get_ts()

            # May need to wrap this in a lock, but let's hold off and see if this
            # is needed.
            self.invoke_actions(event.name, timestamp)
            if self.experiment_state == 'paused':
                # If the experiment is paused, don't do anything.
                return
            getattr(self, f'handle_{self.trial_state.name}')(event, timestamp)

    def start_event_timer(self, duration, event):
        # We call the timer `experiment_state` to ensure that it properly ends
        # any existing event-based timers.
        if isinstance(duration, str):
            duration = self.context.get_value(duration)
        log.info('Starting timer for {} with duration {}'.format(event, duration))
        callback = partial(self.handle_event, event)
        self.start_timer('event', duration, callback, cancel_existing=True)

    def stop_event_timer(self):
        self.stop_timer('event')
