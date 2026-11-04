'''
Verbatim copy of NAFCTrialManager as of commit db7952e (before M0 hardening).

Used only by the golden test that checks the hardened NAFCTrialManager
produces identical trial sequences when no trial is re-prepared. Do not
modify.
'''
import numpy as np

from psibehavior.paradigms.trial_manager_plugin import BaseTrialManager


class LegacyNAFCTrialManager(BaseTrialManager):

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

    def prepare_trial(self):
        repeat_mode = self.get_value('repeat_incorrect')
        target_probability = self.get_value('target_probability')
        n_remind = self.get_value('remind_trials')
        n_warmup = self.get_value('warmup_trials')
        max_reference = self.get_value('max_reference')

        if self.trial_number < n_remind:
            # String of consecutive easy target trials at beginning to get
            # animal warmed up.
            trial_subtype = 'remind'
            self.current_stim = self.next_stim(remind=True, target_probability=1)
        elif self.controller._remind_requested:
            # User has requested an easy target trial.
            trial_subtype = 'remind'
            self.current_stim = self.next_stim(remind=True, target_probability=1)
            self.controller._remind_requested = False
        elif self.trial_number < (n_remind + n_warmup):
            # Now, continue providing easy trials, but alternate between target
            # and reference.
            trial_subtype = 'remind'
            self.current_stim = self.next_stim(remind=True, target_probability=target_probability)
        elif self.consecutive_reference > max_reference:
            # We have exceeded the maximum number of reference trials. Force a
            # target trial.
            trial_subtype = 'forced'
            self.current_stim = self.next_stim(target_probability=1)
        elif self.prior_response is None:
            # No remind or warmup trials have been requested, and this is the
            # very first trial. Just go right into the task.
            trial_subtype = None
            self.current_stim = self.next_stim(target_probability=target_probability)
        elif repeat_mode == 1 and self.prior_response == 'early_np':
            # Repeat only trials where animal did not maintain nose poke long
            # enough (e.g., during the nose poke hold period).
            trial_subtype = 'repeat'
        elif repeat_mode == 2 and self.prior_score != self.controller.scores.correct:
            # Repeat all incorrect trials.
            trial_subtype = 'repeat'
        elif repeat_mode == 3 and \
            self.prior_score != self.controller.scores.correct and \
            self.current_stim['trial_type'] == 'reference':
            # Repeat only nogo trials where the animal false alarmed.
            trial_subtype = 'repeat'
        else:
            # This is not the first trial, it is outside the remind/warmup
            # window, and does not need to be repeated.
            trial_subtype = None
            self.current_stim = self.next_stim(target_probability=target_probability)

        self.current_stim['trial_subtype'] = trial_subtype
        trial_type = self.current_stim['trial_type']
        if trial_type == 'reference':
            self.consecutive_reference += 1
        else:
            self.consecutive_reference = 0

        self.context.set_values(self.current_stim)
        self.controller.trial_state_str = self.get_trial_state_str(self.current_stim)

        waveform = self.stim_waveform(self.current_stim)
        with self.output.engine.lock:
            self.waveform = self.output.add_waveform(waveform)
        return self.get_conditions(self.current_stim)

    def start_trial(self, delay):
        with self.output.engine.lock:
            ts = self.controller.get_ts()
            self.waveform.start(ts + delay, allow_belated=True)
        return ts

    def end_trial(self, delay=0):
        with self.output.engine.lock:
            self.waveform.cancel()

    def trial_complete(self, response, score, info):
        info['stim_start'] = self.waveform.actual_ts
        info['stim_complete'] = self.waveform.complete
        self.prior_response = response
        self.prior_score = score
        self.trial_number += 1

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

        if not self.deliver_reward(stim):
            # Override reward condition to not provide a reward if it's near
            # threshold.
            reward_condition = []
        return response_condition, reward_condition, timeout_condition

    def next_stim(self, target_probability, remind=False):
        raise NotImplementedError

    def deliver_reward(self, stim):
        raise NotImplementedError

    def get_stim_str(self, stim):
        raise NotImplementedError

    def stim_waveform(self, stim):
        raise NotImplementedError
