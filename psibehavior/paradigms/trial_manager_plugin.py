import numpy as np


class BaseTrialManager:

    #: List of parameters to add to GUI. Eliminates the need for a custom
    #: `TrialManagerManifest` class if you only want to add new parameters to
    #: the GUI.
    default_parameters = []

    #: Default group name for each parameter in `default_parameters`. If None,
    #: `group_name`, must be provided for each parameter.
    default_group_name = None

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


GONOGO_PARAMETERS = [
    {
        'name': 'go_probability',
        'label': 'Go probability (frac.)',
        'compact_label': 'Pr',
        'default': 0.5,
        'scope': 'arbitrary',
    },
    {
        'name': 'max_nogo',
        'label': 'Max. consecutive nogo trials',
        'compact_label': 'Max. NG',
        'default': 5,
        'scope': 'arbitrary',
    },
    {
        'name': 'remind_trials',
        'label': 'Remind trials',
        'compact_label': 'N remind',
        'scope': 'experiment',
        'default': 10,
    },
    {
        'name': 'warmup_trials',
        'label': 'Warmup trials',
        'compact_label': 'N warmup',
        'scope': 'experiment',
        'default': 20,
    },
]


class GoNogoTrialManager(BaseTrialManager):

    default_parameters = GONOGO_PARAMETERS

    def __init__(self, controller, prefix=''):
        super().__init__(controller, prefix)
        # Attributes used to track state of trials
        self.prior_info = None, None
        self.trial_type = None
        self.trial_subtype = None
        self.rng = np.random.default_rng()
        self.consecutive_nogo = 0
        self.current_stim = {}

    def next_ttype(self):
        '''
        Determine next trial type.
        '''
        n_remind = self.get_value('remind_trials')
        n_warmup = self.get_value('warmup_trials')
        go_probability = self.get_value('go_probability')
        repeat_mode = self.get_value('repeat_incorrect')
        max_nogo = self.get_value('max_nogo')

        if self.trial_number < n_remind:
            return ('go', 'remind')

        if self.controller._remind_requested:
            return ('go', 'remind')
            self.controller._remind_requested = False

        if self.trial_number < (n_remind + n_warmup):
            return ('go', 'remind') \
                if self.rng.uniform() <= go_probability else ('nogo', None)

        if self.consecutive_nogo >= max_nogo:
            return ('go', 'forced')

        if repeat_mode != 0:
            ttype, resp, score = self.prior_info
            if repeat_mode == 1 and resp == 'early_np':
                # early withdraw only
                return (ttype, 'repeat')
            if repeat_mode == 2 and score != self.controller.scores.correct:
                # all incorrect trials
                return (ttype, 'repeat')
            if repeat_mode == 3 and ttype == 'nogo' and \
                    score != self.controller.scores.correct:
                # Only FA trials
                return (ttype, 'repeat')

        return ('go', None) \
            if self.rng.uniform() <= go_probability else ('nogo', None)

    def prepare_trial(self):
        self.trial_type, self.trial_subtype = self.next_ttype()
        self.current_stim = self.next_stim(self.trial_type, self.trial_subtype)

        # Update context with some information about trial and stim
        self.context.set_value('trial_type', self.trial_type)
        self.context.set_value('trial_subtype', self.trial_subtype)
        self.context.set_values(self.current_stim)

        # Prepare a string representation for GUI
        stim_info = self.stim_info(self.current_stim)
        ttype_info = f'Trial {self.trial_number + 1}: {self.trial_type}'
        if self.trial_subtype is not None:
            ttype_info = f'{ttype_info} ({self.trial_subtype})'
        self.controller.trial_state_str = f'{ttype_info}{stim_info}'

        waveform = self.stim_waveform(self.current_stim)
        with self.output.engine.lock:
            self.output.set_waveform(waveform)

        if self.trial_type.startswith('nogo'):
            return [0], [0], [1]
        else:
            return [1], [1], [0]

    def start_trial(self, delay):
        with self.output.engine.lock:
            ts = self.controller.get_ts()
            self.output.start_waveform(ts + delay)
        return ts

    def end_trial(self, delay=0):
        with self.output.engine.lock:
            ts = self.controller.get_ts()
            self.output.stop_waveform(ts + delay)

    def trial_complete(self, response, score, info):
        self.prior_info = self.trial_type, response, score
        self.trial_number += 1
        if self.trial_type.startswith('nogo'):
            self.consecutive_nogo += 1
        else:
            self.consecutive_nogo = 0

    def next_stim(self, trial_type):
        raise NotImplementedError

    def stim_info(self, stim):
        raise NotImplementedError

    def stim_waveform(self, stim):
        raise NotImplementedError


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
            self.output.set_waveform(waveform)
        return self.get_conditions(self.current_stim)

    def start_trial(self, delay):
        with self.output.engine.lock:
            ts = self.controller.get_ts()
            self.output.start_waveform(ts + delay)
        return ts

    def end_trial(self, delay=0):
        with self.output.engine.lock:
            ts = self.controller.get_ts()
            self.output.stop_waveform(ts + delay)

    def trial_complete(self, response, score, info):
        self.prior_response = response
        self.prior_score = score
        self.trial_number += 1
