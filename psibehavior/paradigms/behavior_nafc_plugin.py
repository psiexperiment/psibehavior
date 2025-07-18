import logging
log = logging.getLogger(__name__)

import enum

from atom.api import Bool, Dict, Int, List, Str, Typed
from enaml.application import timed_call
from enaml.core.api import d_
import numpy as np

#from psilbhb.stim.wav_set import WavSet
from .base_behavior_plugin import (BaseBehaviorPlugin, TrialState)
from .trial_manager_plugin import (BaseTrialManager, BaseContinuousStimManager)


################################################################################
# Supporting
################################################################################
class NAFCTrialScore(enum.Enum):
    '''
    Defines the different types of scores for each trial in a behavioral
    experiment
    '''
    invalid = 0
    incorrect = 1
    correct = 2


class NAFCTrialState(TrialState):
    '''
    Defines the possible states that the experiment can be in. We use an Enum to
    minimize problems that arise from typos by the programmer (e.g., they may
    accidentally set the state to "waiting_for_timeout" rather than
    "waiting_for_to").
    '''
    waiting_for_resume = 'waiting for resume'
    waiting_for_trial_start = 'waiting for trial start'
    waiting_for_np_start = 'waiting for nose-poke start'
    waiting_for_np_duration = 'waiting for nose-poke duration'
    waiting_for_hold = 'waiting for hold'
    waiting_for_response = 'waiting for response'
    waiting_for_to = 'waiting for timeout'
    waiting_for_iti = 'waiting for intertrial interval'
    waiting_for_np_end = 'waiting for nose-poke end'
    waiting_for_reward_end = 'waiting for reward contact'


class BaseNAFCEvent(enum.Enum):
    '''
    Defines the possible events that may occur during the course of the
    experiment.
    '''
    @property
    def category(self):
        return self.value[0]

    @property
    def phase(self):
        return self.value[1]

    @property
    def side(self):
        return self.value[2]

    @property
    def user_initiated(self):
        return self.value[3]


################################################################################
# Plugin
################################################################################
class BehaviorPlugin(BaseBehaviorPlugin):
    '''
    Plugin for controlling appetitive experiments that are based on a reward.
    Eventually this should become generic enough that it can be used with
    aversive experiments as well (it may already be sufficiently generic).
    '''
    #: True if the user is controlling when the trials begin (e.g., in
    #: training).
    manual_control = d_(Bool(), writable=False)

    #: True if we're running in random behavior mode for debugging purposes,
    #: False otherwise.
    random_behavior_mode = Bool(False)

    #: Used internally to track next trial state for after when reward contact
    #: ends (e.g., go to timeout or to intertrial interval)?
    next_trial_state = Str()

    #: Number of response options (e.g., 1 for go-nogo and 2 for 2AFC). Code is
    #: written such that we can eventually add additional choices. Thsese
    #: should typically be set via the plugin factory generator e.g., see
    #: `behavior_nafc.enaml`.).
    N_response = Int(1)

    #: What to call the response. Included for backwards-compatibility with
    #: psilbhb which expects to see `spout` instead of `response`.
    response_name = Str('response')

    #: Mapping of the rising/falling edges detected on digital channels to an event.
    event_map = Dict()

    #: True if nose-poke is active
    np_active = Bool(False)

    #: String provided by the trial_manager plugin that generates a UI message
    #: regarding the trial.
    trial_state_str = Str('')

    #: Trial object
    trial_manager = Typed(BaseTrialManager)

    #: Stim managers
    stim_managers = List(Typed(BaseContinuousStimManager))

    #: Enum indicating appropriate response. If -1, a response on any reward
    #: port is acceptable and should be rewarded. If 0, a response on any port
    #: is not acceptable and should not be rewarded. Greater than 0 indicates
    #: that the particular numbered port is the correct response.
    response_condition = Int(0)

    #: This needs to be dynamically generated since the events will depend on
    #: `N_response`.
    events = Typed(enum.EnumType)

    scores = NAFCTrialScore

    def _default_events(self):
        members = {
            'hold_start': ('hold', 'start', -1, False),
            'hold_duration_elapsed': ('hold', 'elapse', -1, False),

            'response_start': ('response', 'start', -1, False),
            'response_end': ('response', 'end', -1, False),
            'response_duration_elapsed': ('response', 'elapsed', -1, False),

            # The third value in the tuple is the response port the animal is
            # responding to.  The fourth value indicates whether it was manually
            # triggered by the user via the GUI (True) or by the animal (False).
            'np_start': ('np', 'start', False, 0, False),
            'np_end': ('np', 'end', False, 0, False),
            'digital_np_start': ('np', 'start', 0, True),
            'digital_np_end': ('np', 'end', 0, True),

            'np_duration_elapsed': ('np', 'elapsed'),
            'to_start': ('to', 'start'),
            'to_end': ('to', 'end'),
            'to_duration_elapsed': ('to', 'elapsed'),

            'iti_start': ('iti', 'start'),
            'iti_end': ('iti', 'end'),
            'iti_duration_elapsed': ('iti', 'elapsed'),

            'trial_start': ('trial', 'start'),
            'trial_end': ('trial', 'end'),
        }

        for i in range(self.N_response):
            # The third value in the tuple is the response port the animal is
            # responding to.  The fourth value indicates whether it was manually
            # triggered by the user via the GUI (True) or by the animal (False).
            members[f'resp_{i+1}_start'] = ('response', 'start', i+1, False)
            members[f'resp_{i+1}_end'] = ('response', 'end', i+1, False)
            members[f'digital_resp_{i+1}_start'] = ('response', 'start', i+1, True)
            members[f'digital_resp_{i+1}_end'] = ('response', 'end', i+1, True)

        return BaseNAFCEvent('NAFCEvent', list(members.items()))

    def handle_event(self, event, timestamp=None):
        if event in (self.events.np_start, self.events.digital_np_start):
            self.np_active = True
        elif event in (self.events.np_end, self.events.digital_np_end):
            self.np_active = False
        super().handle_event(event, timestamp)

    def request_trial(self):
        self.prepare_trial(auto_start=True)

    def _default_trial_state(self):
        return NAFCTrialState.waiting_for_resume

    def _default_event_map(self):
        event_map = {
            ('rising', 'np_contact'): self.events.np_start,
            ('falling', 'np_contact'): self.events.np_end,
        }
        for i in range(self.N_response):
            event_map['rising', f'{self.response_name}_contact_{i+1}'] = \
                getattr(self.events, f'resp_{i+1}_start')
            event_map['falling', f'{self.response_name}_contact_{i+1}'] = \
                getattr(self.events, f'resp_{i+1}_end')
        return event_map

    def can_modify(self):
        return self.trial_state in (
            NAFCTrialState.waiting_for_resume,
            NAFCTrialState.waiting_for_trial_start,
            NAFCTrialState.waiting_for_np_start,
            NAFCTrialState.waiting_for_iti,
            NAFCTrialState.waiting_for_reward_end,
        )

    def prepare_trial(self, auto_start=False):
        log.info('Preparing for next trial (auto_start %r)', auto_start)
        # Figure out next trial and set up selector.
        self.manual_control = self.context.get_value('manual_control')
        self.trial_info = {
            'response_start': np.nan,
            'response_ts': np.nan,
        }
        self.trial_state = NAFCTrialState.waiting_for_trial_start

        #: TODO: Allow for capturing additional info here.
        self.response_condition = self.trial_manager.prepare_trial()

        # Now trigger any callbacks that are listening for the trial_ready
        # event.
        self.invoke_actions('trial_ready')
        if auto_start:
            self.start_trial()
        elif self.np_active:
            # If animal is already poking and ITI is over, go ahead and start
            # trial.
            self.trial_info['np_start'] = self.get_ts()
            self.start_trial()
        else:
            self.trial_state = NAFCTrialState.waiting_for_np_start

    def handle_waiting_for_trial_start(self, event, timestamp):
        pass

    def handle_waiting_for_np_start(self, event, timestamp):
        if self.experiment_state != 'running':
            return
        if self.manual_control:
            return

        if event.category == 'np' and event.phase == 'start':
            # Animal has nose-poked in an attempt to initiate a trial.
            self.trial_state = NAFCTrialState.waiting_for_np_duration
            self.start_event_timer('np_duration', self.events.np_duration_elapsed)
            # If the animal does not maintain the nose-poke long enough,
            # this value will be deleted.
            self.trial_info['np_start'] = timestamp

    def handle_waiting_for_np_duration(self, event, timestamp):
        if event.category == 'np' and event.phase == 'end':
            # Animal has withdrawn from nose-poke too early. Cancel the timer
            # so that it does not fire a 'event_np_duration_elapsed'.
            log.debug('Animal withdrew too early')
            self.stop_event_timer()
            self.trial_state = NAFCTrialState.waiting_for_np_start
            del self.trial_info['np_start']
        elif event.category == 'np' and event.phase == 'elapsed':
            log.debug('Animal initiated trial')
            self.start_trial()

    def start_trial(self):
        log.info('Starting next trial')
        target_delay = self.context.get_value('target_delay')
        # This is broken into a separate method from
        # handle_waiting_for_np_duration to allow us to trigger this method
        # from a toolbar button for training purposes.
        ts = self.trial_manager.start_trial(target_delay)
        self.invoke_actions('trial_start', ts)
        self.trial_info['trial_start'] = ts
        # Notify the state machine that we are now in the hold phase of trial.
        # This means that the next time any event occurs (e.g., such as one
        # detected on the digital lines), it will call
        # `handle_waiting_for_hold` with the event and timestamp the event
        # occurred at.

        # First, check to see if we expect the animal to hold the nose-poke for
        # a certain duration. If not, then go straight to the response phase.
        hold_duration = self.context.get_value('hold_duration')
        if hold_duration != 0:
            self.advance_state('hold', ts)
        else:
            self.advance_state('response', ts)

    def handle_waiting_for_hold(self, event, timestamp):
        if event.category == 'np' and event.phase == 'end':
            # If animal withdraws during NP hold period, end trial.
            self.invoke_actions('response_end', timestamp)
            self.trial_info['response_ts'] = timestamp
            self.trial_info['response_side'] = np.nan
            self.end_trial('early_np', self.scores.invalid)
        elif event == self.events.hold_duration_elapsed:
            log.info('Hold duration over')
            self.advance_state('response', timestamp)

    def handle_waiting_for_response(self, event, timestamp):
        # Event is a tuple of 'response', 'start', side, True/False where False
        # indicates animal initiated event and True indicates human initiated
        # event via button. See the definbition of the NAFCEvent enum.
        log.info(f'Waiting for response. Received {event} at {timestamp}')

        if event == self.events.response_start:
            self.trial_info['response_start'] = timestamp
            # If we are in training mode, deliver a reward preemptively
            if self.context.get_value('training_mode') and (self.response_condition > 0):
                self.invoke_actions(f'deliver_reward_{self.response_condition}', timestamp)
            return

        if self.N_response == 1:
            # This is a special-case section for scoring go-nogo, which is
            # defined when the number of response inputs are 1. A repoke into
            # the nose port or no response will be scored as a "no" response
            # (i.e., the subject did not hear the target). A response at the
            # single response input will be socred as a "yes" response.
            if event.category == 'np' and event.phase == 'start':
                self.trial_info['response_ts'] = timestamp
                self.trial_info['response_side'] = 0
                response = 'np'
                score = self.scores.correct if self.response_condition == 0 \
                    else self.scores.incorrect
            elif event.category == 'response' and event.phase == 'elapsed':
                self.trial_info['response_ts'] = np.nan
                self.trial_info['response_side'] = np.nan
                response = 'no_response'
                score = self.scores.correct if self.response_condition == 0 \
                    else self.scores.incorrect
            elif event.category == 'response' and event.phase == 'start':
                self.trial_info['response_ts'] = timestamp
                self.trial_info['response_side'] = event.side
                response = f'{self.response_name}_{event.side}'
                score = self.scores.correct if self.response_condition == 1 \
                    else self.scores.incorrect
                if not self.context.get_value('training_mode'):
                    self.invoke_actions(f'deliver_reward_{event.side}', timestamp)
            else:
                # This event does not need to be handled. Ignore and bypass any
                # additional logic.
                return
            self.invoke_actions('response_end', timestamp)
            self.end_trial(response, score)

        else:
            # This is the NAFC section of the scoring.
            if event.category == 'response' and event.phase == 'start':
                self.invoke_actions('response_end', timestamp)
                self.trial_info['response_ts'] = timestamp
                self.trial_info['response_side'] = event.side
                if self.response_condition == -1:
                    # This is a flag indicating that any of the response ports
                    # can be the correct one.
                    score = self.scores.correct
                    if not self.context.get_value('training_mode'):
                        self.invoke_actions(f'deliver_reward_{event.side}', timestamp)
                elif self.trial_info['response_side'] == self.response_condition:
                    score = self.scores.correct
                    # If we are in training mode, the reward has already been
                    # delivered.
                    if not self.context.get_value('training_mode'):
                        self.invoke_actions(f'deliver_reward_{event.side}', timestamp)
                else:
                    score = self.scores.incorrect
                response = f'{self.response_name}_{event.side}'
                self.end_trial(response, score)
            elif event.category == 'response' and event.phase == 'elapsed':
                self.invoke_actions('response_end', timestamp)
                self.trial_info['response_ts'] = np.nan
                self.trial_info['response_side'] = np.nan
                self.end_trial('no_response', self.scores.invalid)

    def end_trial(self, response, score):
        self.stop_event_timer()
        ts = self.get_ts()
        log.info(f'Ending trial with {response} scored as {score}')

        response_time = self.trial_info['response_ts']-self.trial_info['trial_start']
        self.trial_info.update({
            'response': response,
            'score': score.name,
            'correct': score == self.scores.correct,
            'response_time': response_time,
        })
        self.trial_info.update(self.context.get_values())
        self.invoke_actions('trial_end', ts, kw={'result': self.trial_info.copy()})
        self.trial_manager.trial_complete(response, score, self.trial_info)

        if score == self.scores.invalid:
            # Early withdraw from nose-poke want to stop sound
            self.trial_manager.end_trial()

        if score in (self.scores.incorrect, self.scores.invalid):
            if self.context.get_value('to_duration') > 0:
                next_state = 'to'
            else:
                next_state = 'iti'
        else:
            next_state = 'iti'

        if next_state == 'to':
            self.invoke_actions('to_start')

        # Check if animal is at reward hopper
        if response.startswith(self.response_name):
            self.trial_state = NAFCTrialState.waiting_for_reward_end
            self.next_trial_state = next_state
        elif not response.startswith('np') and self.np_active and not self.context.get_value('continuous_np'):
            self.trial_state = NAFCTrialState.waiting_for_np_end
            self.next_trial_state = next_state
        elif next_state == 'to':
            self.start_event_timer(f'to_duration', self.events.to_duration_elapsed)
            self.trial_state = NAFCTrialState.waiting_for_to
        else:
            self.advance_state('iti', ts)

        # Apply pending changes that way any parameters (such as repeat_FA or
        # go_probability) are reflected in determining the next trial type.
        if self._apply_requested:
            self.apply_changes()

    def advance_state(self, state, timestamp):
        log.info(f'Advancing to {state}')
        self.trial_state = getattr(NAFCTrialState, f'waiting_for_{state}')
        elapsed_event = getattr(self.events, f'{state}_duration_elapsed')
        self.start_event_timer(f'{state}_duration', elapsed_event)
        start_event = getattr(self.events, f'{state}_start')
        self.handle_event(start_event, timestamp)

    def handle_waiting_for_np_end(self, event, timestamp):
        if event.category == 'np' and event.phase == 'end':
            if self.next_trial_state == 'to':
                self.start_event_timer(f'to_duration', self.events.to_duration_elapsed)
                self.trial_state = NAFCTrialState.waiting_for_to
            elif self.next_trial_state == 'iti':
                self.advance_state('iti', timestamp)

    def handle_waiting_for_reward_end(self, event, timestamp):
        if event.category == 'response' and event.phase == 'end':
            if self.next_trial_state == 'to':
                self.start_event_timer(f'to_duration', self.events.to_duration_elapsed)
                self.trial_state = NAFCTrialState.waiting_for_to
            elif self.next_trial_state == 'iti':
                self.advance_state('iti', timestamp)

    def handle_waiting_for_to(self, event, timestamp):
        if event == self.events.to_duration_elapsed:
            # Process end of timeout (e.g., turn lights back on).
            self.invoke_actions('to_end', timestamp)
            self.advance_state('iti', timestamp)
        elif event.category == 'response' and event.phase == 'start':
            # Cancel timeout timer and wait for animal to disconnect from
            # response port.
            self.stop_event_timer()
            self.next_trial_state = 'to'
            self.trial_state = NAFCTrialState.waiting_for_reward_end

    def handle_waiting_for_iti(self, event, timestamp):
        if event.category == 'response' and event.phase == 'start':
            # Animal attempted to get reward. Reset ITI interval.
            self.stop_event_timer()
            self.next_trial_state = 'iti'
            self.trial_state = NAFCTrialState.waiting_for_reward_end
        elif event == self.events.iti_duration_elapsed:
            self.invoke_actions(self.events.iti_end.name, timestamp)
            if self._pause_requested:
                self.pause_experiment()
                self.trial_state = NAFCTrialState.waiting_for_resume
            elif self.experiment_state != 'running':
                pass
            else:
                self.prepare_trial()

    def start_random_behavior(self):
        log.info('Starting random behavior mode')
        self.random_behavior_mode = True
        timed_call(500, self.random_behavior_cb, self.events.digital_np_start)

    def stop_random_behavior(self):
        self.random_behavior_mode = False

    def random_behavior_cb(self, event):
        if self.random_behavior_mode:
            log.info('Handling event %r', event)
            self.handle_event(event)
            ms = np.random.uniform(100, 3000)
            if event == self.events.digital_np_start:
                next_event = self.events.digital_np_end
            else:
                next_event = self.events.digital_np_start
            log.info('Starting next event for %d ms from now', ms)
            timed_call(ms, self.random_behavior_cb, next_event)
