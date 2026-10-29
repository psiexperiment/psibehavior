import logging
log = logging.getLogger(__name__)

from dataclasses import dataclass

import numpy as np


class BaseTrialManager:

    #: List of parameters to add to GUI. Eliminates the need for a custom
    #: `TrialManagerManifest` class if you only want to add new parameters to
    #: the GUI.
    default_parameters = []

    #: Default group name for each parameter in `default_parameters`. If None,
    #: `group_name`, must be provided for each parameter.
    default_group_name = None

    #: EpochWaveform instance used for managing waveform.
    waveform = None

    @classmethod
    def get_default_parameters(cls, **kwargs):
        '''
        Return the parameters to add to the GUI.

        `kwargs` are the `manager_kwargs` set on the `TrialManagerManifest`.
        Subclasses can override this to tailor the parameters to the paradigm
        definition. Unknown keyword arguments must be ignored. The result must
        be deterministic since the generated context items are cached per
        manifest.
        '''
        return cls.default_parameters

    def __init__(self, controller, prefix=''):
        self.controller = controller
        self.context = controller.context
        self.prefix = prefix
        self.trial_number = 0

    def get_value(self, value_name, *args, **kw):
        '''
        Get value of parameter, applying prefix if needed
        '''
        return self.context.get_value(f'{self.prefix}{value_name}', *args, **kw)

    def prepare_trial(self):
        '''
        Subclasses are responsible for setting 'trial_type' and 'trial_subtype'
        in the context, e.g.:

            self.context.set_value('trial_type', 'nogo')
            self.context.set_value('trial_subtype', 'repeat')

        Trial type will be specific to the paradigm and should be coded
        appropriately. Trial subtype indicates if it is a special class (e.g.,
        'repeat', 'forced', 'remind', 'warmup') that may need to be excluded
        from analysis.

        This may be called more than once before the trial is started (e.g.,
        when the user applies changes or requests a remind trial). Any state
        that tracks the trial sequence should be updated in `trial_complete`,
        not here.

        Returns
        -------
        int : correct response port
            If greater than 0, indicates response port that is rewarded
            (response ports are numbered starting at 1). If 0, this indicates
            this is a nogo trial (i.e., no reward). If -1, indicates any
            response port recieves a reward. For go/nogo experiments, there is
            only one response port which will be numbered 1.
        '''
        raise NotImplementedError

    def start_trial(self, delay):
        raise NotImplementedError

    def end_trial(self, delay):
        raise NotImplementedError

    def trial_complete(self, response, score, info):
        raise NotImplementedError


@dataclass
class PreparedTrial:
    '''
    A trial that has been prepared by `NAFCTrialManager.prepare_trial`.
    '''
    #: Stimulus (and trial type/subtype) that will be written to the context.
    stim: dict

    #: Waveform queued on the output for this trial.
    waveform: object

    #: Everything the plan depended on. If this is unchanged at the next
    #: `prepare_trial`, the trial is reused rather than re-planned.
    key: tuple

    #: Why this trial was chosen (e.g., 'n_remind', 'user_remind', 'warmup',
    #: 'forced', 'repeat', 'normal').
    source: str

    #: True once `start_trial` has been called.
    started: bool = False


class NAFCTrialManager(BaseTrialManager):

    default_parameters = [
        {
            'name': 'target_probability',
            'label': 'Target probability (frac.)',
            'compact_label': 'Pr',
            'default': 0.5,
            'scope': 'arbitrary',
            'group_name': 'trial_sequence',
        },
        {
            'name': 'max_reference',
            'label': 'Max. consec. ref. trials',
            'compact_label': 'Max. Ref.',
            'default': 5,
            'scope': 'arbitrary',
            'group_name': 'trial_sequence',
        },
        {
            'name': 'remind_trials',
            'label': 'Remind trials',
            'compact_label': 'N remind',
            'scope': 'experiment',
            'default': 10,
            'group_name': 'trial_sequence',
        },
        {
            'name': 'warmup_trials',
            'label': 'Warmup trials',
            'compact_label': 'N warmup',
            'scope': 'experiment',
            'default': 20,
            'group_name': 'trial_sequence',
        },
    ]

    def __init__(self, controller, prefix=''):
        super().__init__(controller, prefix)
        self.prior_response = None
        self.prior_score = None
        self.rng = np.random.default_rng()
        self.consecutive_reference = 0
        self.trial_number = 0

        #: Stimulus of the most recently prepared trial. Kept for subclasses
        #: that read it; use `last_completed_stim` for the previous trial.
        self.current_stim = {}

        #: Stimulus of the most recently completed trial.
        self.last_completed_stim = None

        #: Trial that has been prepared but not started.
        self._pending = None

        #: Trial that has been started but not completed.
        self._played = None

        #: True once we have warned that a subclass rewards reference trials
        #: via `deliver_reward`.
        self._warned_reference_reward = False

    ############################################################################
    # Trial lifecycle
    ############################################################################
    def prepare_trial(self):
        '''
        Prepare the next trial, or reuse the prepared trial if nothing it
        depends on has changed.

        The trial sequence state (trial number, consecutive references,
        prior response) is only updated in `trial_complete`, so calling this
        repeatedly before the trial starts neither re-draws the stimulus nor
        double-counts anything.
        '''
        remind_requested = bool(self.controller._remind_requested)
        key = (self._snapshot(), remind_requested, self.trial_number)

        pending = self._pending
        if pending is not None and not pending.started:
            if pending.key == key:
                log.info('Reusing prepared trial')
                return self._activate(pending)
            log.info('Discarding prepared trial')
            with self.output.engine.lock:
                pending.waveform.cancel()

        stim, source = self._plan(remind_requested)
        waveform = self.stim_waveform(stim)
        with self.output.engine.lock:
            waveform = self.output.add_waveform(waveform)
        pending = PreparedTrial(stim=stim, waveform=waveform, key=key,
                                source=source)
        self._pending = pending
        self.current_stim = stim
        return self._activate(pending)

    def start_trial(self, delay):
        pending = self._pending
        if pending is None or pending.started:
            raise RuntimeError('No prepared trial to start')
        with self.output.engine.lock:
            ts = self.controller.get_ts()
            pending.waveform.start(ts + delay, allow_belated=True)
        pending.started = True
        self._played = pending
        self.waveform = pending.waveform
        if pending.source == 'user_remind':
            # Clear the request once the remind trial actually starts. If it
            # were cleared when the trial was prepared, a re-prepare (e.g.,
            # Apply) before the trial starts would lose the request.
            self.controller._remind_requested = False
        return ts

    def end_trial(self, delay=0):
        trial = self._played if self._played is not None else self._pending
        if trial is None:
            return
        with self.output.engine.lock:
            trial.waveform.cancel()

    def trial_complete(self, response, score, info):
        played = self._played
        if played is None:
            # Should never happen. Keep the trial log columns consistent and
            # leave the trial sequence untouched.
            log.error('Trial completed without a started trial')
            info['stim_start'] = np.nan
            info['stim_complete'] = False
            return
        info['stim_start'] = played.waveform.actual_ts
        info['stim_complete'] = played.waveform.complete
        self.prior_response = response
        self.prior_score = score
        self.last_completed_stim = played.stim
        if played.stim['trial_type'] == 'reference':
            self.consecutive_reference += 1
        else:
            self.consecutive_reference = 0
        self.trial_number += 1
        if self._pending is played:
            self._pending = None
        self._played = None

    ############################################################################
    # Planning
    ############################################################################
    def _snapshot(self):
        '''
        Values of the parameters that the trial plan depends on.

        This covers the parameters owned by the trial manager (including the
        stimulus parameters defined by subclasses) plus `repeat_incorrect`. If
        any of these change (e.g., the user applies a change to the stimulus
        parameters), the prepared trial is discarded and re-planned.
        '''
        names = [f'{self.prefix}{p["name"]}' for p in self.default_parameters
                 if p.get('type', 'Parameter') != 'Result']
        names.append('repeat_incorrect')
        values = []
        for name in names:
            try:
                values.append(repr(self.context.get_value(name)))
            except Exception as e:
                values.append(f'<{type(e).__name__}>')
        return tuple(values)

    def _plan(self, remind_requested):
        '''
        Choose the next trial using only the state of completed trials.

        Returns
        -------
        stim : dict
            Stimulus returned by `next_stim` (or copied from the last completed
            trial for a repeat) with `trial_subtype` added.
        source : str
            Why this trial was chosen.
        '''
        repeat_mode = self.context.get_value('repeat_incorrect')
        target_probability = self.get_value('target_probability')
        n_remind = self.get_value('remind_trials')
        n_warmup = self.get_value('warmup_trials')
        max_reference = self.get_value('max_reference')
        correct = self.controller.scores.correct

        if self.trial_number < n_remind:
            # String of consecutive easy target trials at beginning to get
            # animal warmed up.
            trial_subtype, source = 'remind', 'n_remind'
            stim = self.next_stim(remind=True, target_probability=1)
        elif remind_requested:
            # User has requested an easy target trial.
            trial_subtype, source = 'remind', 'user_remind'
            stim = self.next_stim(remind=True, target_probability=1)
        elif self.trial_number < (n_remind + n_warmup):
            # Now, continue providing easy trials, but alternate between target
            # and reference.
            trial_subtype, source = 'remind', 'warmup'
            stim = self.next_stim(remind=True, target_probability=target_probability)
        elif self.consecutive_reference > max_reference:
            # We have exceeded the maximum number of reference trials. Force a
            # target trial.
            trial_subtype, source = 'forced', 'forced'
            stim = self.next_stim(target_probability=1)
        elif self.prior_response is None or self.last_completed_stim is None:
            # No remind or warmup trials have been requested, and this is the
            # very first trial. Just go right into the task.
            trial_subtype, source = None, 'normal'
            stim = self.next_stim(target_probability=target_probability)
        elif repeat_mode == 1 and self.prior_response == 'early_np':
            # Repeat only trials where animal did not maintain nose poke long
            # enough (e.g., during the nose poke hold period).
            trial_subtype, source = 'repeat', 'repeat'
            stim = self.last_completed_stim
        elif repeat_mode == 2 and self.prior_score != correct:
            # Repeat all incorrect trials.
            trial_subtype, source = 'repeat', 'repeat'
            stim = self.last_completed_stim
        elif repeat_mode == 3 and self.prior_score != correct and \
                self.last_completed_stim['trial_type'] == 'reference':
            # Repeat only nogo trials where the animal false alarmed.
            trial_subtype, source = 'repeat', 'repeat'
            stim = self.last_completed_stim
        else:
            # This is not the first trial, it is outside the remind/warmup
            # window, and does not need to be repeated.
            trial_subtype, source = None, 'normal'
            stim = self.next_stim(target_probability=target_probability)

        stim = dict(stim)
        stim['trial_subtype'] = trial_subtype
        return stim, source

    def _activate(self, trial):
        # Always (re)write the stimulus to the context. Applying changes
        # clears values set in the context, so a reused trial must be written
        # again.
        self.context.set_values(trial.stim)
        self.controller.trial_state_str = self.get_trial_state_str(trial.stim)
        return self.get_conditions(trial.stim)

    ############################################################################
    # Conditions
    ############################################################################
    def get_trial_state_str(self, stim):
        trial_info = f'Trial {self.trial_number+1}: {stim["trial_type"].capitalize()}'
        if stim['trial_subtype'] is not None:
            trial_info = f'{trial_info} ({stim["trial_subtype"]})'
        stim_info = self.get_stim_str(stim)
        return f'{trial_info}, {stim_info}'

    def get_conditions(self, stim):
        if self.controller.N_response == 2:
            side = 1 if stim['trial_type'] == 'reference' else 2
            response_condition = [side]
            reward_condition = [side]
            timeout_condition = [1, 2]
            timeout_condition.remove(side)
        elif self.controller.N_response == 1:
            if stim['trial_type'] == 'reference':
                response_condition = [-1, 0]
                reward_condition = []
                timeout_condition = [1]
            else:
                response_condition = [1]
                reward_condition = [1]
                timeout_condition = []

        if stim['trial_type'] == 'reference':
            if not self.reward_reference(stim):
                reward_condition = []
        elif not self.deliver_reward(stim):
            # Override reward condition to not provide a reward if it's near
            # threshold.
            reward_condition = []
        return response_condition, reward_condition, timeout_condition

    def reward_reference(self, stim):
        '''
        True if a correct response on a reference trial should be rewarded.

        Only applies when there is a response port for reference trials
        (N_response > 1) and is controlled by `reward_correct_reference`.
        '''
        if self.controller.N_response == 1:
            return False
        if self.context.get_value('reward_correct_reference'):
            return True
        # Before `reward_correct_reference` existed, `deliver_reward` was also
        # applied to reference trials. Honor subclasses that still reward
        # reference trials that way (e.g., ModulationTask with
        # `max_reward_depth` >= 0) so existing settings behave as before.
        if self.deliver_reward(stim):
            if not self._warned_reference_reward:
                log.warning('deliver_reward returned True for a reference '
                            'trial. Correct reference responses will be '
                            'rewarded. Use reward_correct_reference instead.')
                self._warned_reference_reward = True
            return True
        return False

    ############################################################################
    # Methods subclasses must implement
    ############################################################################
    def next_stim(self, target_probability, remind=False):
        raise NotImplementedError

    def deliver_reward(self, stim):
        '''
        True if a correct response on a target trial should be rewarded (e.g.,
        return False for targets near threshold).
        '''
        raise NotImplementedError

    def get_stim_str(self, stim):
        raise NotImplementedError

    def stim_waveform(self, stim):
        raise NotImplementedError
