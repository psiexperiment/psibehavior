'''
How BehaviorPlugin scores each response and where the state machine goes
next (reward contact, nose-poke release, timeout or intertrial interval).

These pin the current behavior, so that refactoring the scoring (e.g., moving
it into the trial manager) can be checked against it. They use the real
plugin and a real context (see `test_behavior_plugin.HarnessPlugin`).

In go/nogo (one response port), a target is rewarded at port 1 and a
reference is withheld from (no response, or a repoke, is correct). In 2AFC,
references are answered at port 1 and targets at port 2.
'''
import numpy as np
import pytest

from psibehavior.paradigms.behavior_nafc_plugin import NAFCTrialState as S

from test_behavior_plugin import make_plugin


#: Settings that make every trial the given type. Level 60 is rewarded (at
#: or above ToyManager's min_reward_level of 30).
TARGET = {'target_probability': 1, 'level_list': [60]}
REFERENCE = {'target_probability': 0, 'level_list': [60]}


def start(plugin):
    '''
    Poke, hold through the nose-poke duration and withdraw. With the default
    hold duration (0), the response window opens as the trial starts.
    '''
    plugin.poke()
    plugin.fire()  # np_duration
    assert plugin.trial_state == S.waiting_for_response
    plugin.h_ts += 0.05
    plugin.withdraw()
    assert plugin.trial_state == S.waiting_for_response


def result(plugin):
    r = plugin.h_results[-1]
    return r['response'], r['score'], r['reward_delivered']


def actions_after(plugin, event):
    actions = plugin.h_actions
    return actions[len(actions) - 1 - actions[::-1].index(event):]


################################################################################
# Go/nogo
################################################################################
def test_gonogo_target_hit():
    plugin = make_plugin(**TARGET)
    start(plugin)
    plugin.respond(1)
    assert result(plugin) == ('resp_1', 'correct', True)
    assert 'deliver_reward_1' in plugin.h_actions
    assert plugin.h_results[-1]['response_side'] == 1
    # Waits for the animal to leave the reward port before the ITI.
    assert plugin.trial_state == S.waiting_for_reward_end
    plugin.release(1)
    assert plugin.trial_state == S.waiting_for_iti
    assert 'to_start' not in plugin.h_actions


def test_gonogo_target_hard_hit_not_rewarded():
    plugin = make_plugin(target_probability=1, level_list=[20])
    start(plugin)
    plugin.respond(1)
    assert result(plugin) == ('resp_1', 'correct', False)
    assert 'deliver_reward_1' not in plugin.h_actions


def test_gonogo_target_miss_is_invalid():
    plugin = make_plugin(**TARGET)
    start(plugin)
    plugin.fire()  # response_duration
    assert result(plugin) == ('no_response', 'invalid', False)
    assert np.isnan(plugin.h_results[-1]['response_side'])
    # No timeout: a miss has no port that could be in timeout_condition.
    assert 'to_start' not in plugin.h_actions
    assert plugin.trial_state == S.waiting_for_iti


def test_gonogo_target_repoke_is_incorrect_without_timeout():
    plugin = make_plugin(**TARGET)
    start(plugin)
    plugin.h_ts += 0.1
    plugin.poke()
    assert result(plugin) == ('np', 'incorrect', False)
    assert plugin.h_results[-1]['response_side'] == 0
    # The nose-poke port (0) is never in timeout_condition, and a repoke does
    # not wait for the nose-poke to end.
    assert 'to_start' not in plugin.h_actions
    assert plugin.trial_state == S.waiting_for_iti


def test_gonogo_reference_withhold_is_correct():
    plugin = make_plugin(**REFERENCE)
    start(plugin)
    plugin.fire()  # response_duration
    assert result(plugin) == ('no_response', 'correct', False)
    assert plugin.trial_state == S.waiting_for_iti


def test_gonogo_reference_repoke_is_correct():
    plugin = make_plugin(**REFERENCE)
    start(plugin)
    plugin.h_ts += 0.1
    plugin.poke()
    assert result(plugin) == ('np', 'correct', False)
    assert plugin.trial_state == S.waiting_for_iti


def test_gonogo_false_alarm_times_out_after_release():
    plugin = make_plugin(**REFERENCE)
    start(plugin)
    plugin.respond(1)
    assert result(plugin) == ('resp_1', 'incorrect', False)
    assert 'to_start' in plugin.h_actions
    assert 'deliver_reward_1' not in plugin.h_actions
    # The timeout starts once the animal leaves the port.
    assert plugin.trial_state == S.waiting_for_reward_end
    plugin.release(1)
    assert plugin.trial_state == S.waiting_for_to
    plugin.h_actions.clear()
    plugin.fire()  # to_duration
    assert plugin.h_actions == ['to_duration_elapsed', 'to_end', 'iti_start']
    assert plugin.trial_state == S.waiting_for_iti


def test_gonogo_false_alarm_without_timeout_duration():
    plugin = make_plugin(to_duration=0, **REFERENCE)
    start(plugin)
    plugin.respond(1)
    assert result(plugin) == ('resp_1', 'incorrect', False)
    assert 'to_start' not in plugin.h_actions
    plugin.release(1)
    assert plugin.trial_state == S.waiting_for_iti


def test_return_to_port_during_timeout_restarts_it():
    plugin = make_plugin(**REFERENCE)
    start(plugin)
    plugin.respond(1)
    plugin.release(1)
    assert plugin.trial_state == S.waiting_for_to
    plugin.respond(1)
    assert plugin.trial_state == S.waiting_for_reward_end
    assert 'event' not in plugin.h_timers
    plugin.release(1)
    assert plugin.trial_state == S.waiting_for_to
    # The full timeout runs again.
    assert plugin.h_timers['event'][0] == plugin.context.get_value('to_duration')


################################################################################
# Early withdrawal and nose-poke release
################################################################################
@pytest.mark.parametrize('N_response', [1, 2])
def test_early_withdrawal_during_hold(N_response):
    plugin = make_plugin(N_response=N_response, hold_duration=0.5, **TARGET)
    plugin.poke()
    plugin.fire()  # np_duration
    assert plugin.trial_state == S.waiting_for_hold
    waveform = plugin.h_output.waveforms[-1]
    plugin.h_ts += 0.1
    plugin.withdraw()
    assert result(plugin) == ('early_np', 'invalid', False)
    r = plugin.h_results[-1]
    assert r['np_end'] == r['response_ts'] == plugin.h_ts
    assert np.isnan(r['response_side'])
    # The stimulus is stopped, and there is no timeout.
    assert waveform.cancelled
    assert 'to_start' not in plugin.h_actions
    assert plugin.trial_state == S.waiting_for_iti


def test_hold_completed_opens_response_window():
    plugin = make_plugin(hold_duration=0.5, **TARGET)
    plugin.poke()
    plugin.fire()  # np_duration
    plugin.fire()  # hold_duration
    assert plugin.trial_state == S.waiting_for_response
    assert 'response_start' in plugin.h_actions


def test_miss_while_still_poking_waits_for_release():
    plugin = make_plugin(**TARGET)
    plugin.poke()
    plugin.fire()  # np_duration
    plugin.fire()  # response_duration, still poking
    assert result(plugin) == ('no_response', 'invalid', False)
    assert plugin.trial_state == S.waiting_for_np_end
    plugin.withdraw()
    assert plugin.trial_state == S.waiting_for_iti


def test_miss_while_still_poking_with_continuous_np():
    plugin = make_plugin(continuous_np=True, **TARGET)
    plugin.poke()
    plugin.fire()  # np_duration
    plugin.fire()  # response_duration, still poking
    assert plugin.trial_state == S.waiting_for_iti


def test_false_alarm_while_still_poking_times_out_after_release():
    plugin = make_plugin(N_response=2, **REFERENCE)
    plugin.poke()
    plugin.fire()  # np_duration
    # Answers at the wrong port (2) without leaving the nose-poke.
    plugin.respond(2)
    assert result(plugin) == ('resp_2', 'incorrect', False)
    assert plugin.trial_state == S.waiting_for_reward_end
    plugin.release(2)
    assert plugin.trial_state == S.waiting_for_to


def test_response_delay_ignores_quick_repoke():
    plugin = make_plugin(response_delay=0.5, **REFERENCE)
    start(plugin)
    np_end = plugin.h_ts
    plugin.h_ts = np_end + 0.2
    plugin.poke()
    assert plugin.trial_state == S.waiting_for_response
    assert plugin.h_results == []
    plugin.withdraw()
    plugin.h_ts = np_end + 0.6
    plugin.poke()
    assert result(plugin) == ('np', 'correct', False)


################################################################################
# 2AFC
################################################################################
@pytest.mark.parametrize('settings, side, rewarded', [
    (TARGET, 2, True),
    # ToyManager rewards a stimulus at or above min_reward_level, and a
    # reference is level 0.
    (REFERENCE, 1, False),
])
def test_2afc_correct(settings, side, rewarded):
    plugin = make_plugin(N_response=2, **settings)
    start(plugin)
    plugin.respond(side)
    assert result(plugin) == (f'resp_{side}', 'correct', rewarded)
    assert (f'deliver_reward_{side}' in plugin.h_actions) == rewarded
    assert 'to_start' not in plugin.h_actions
    plugin.release(side)
    assert plugin.trial_state == S.waiting_for_iti


@pytest.mark.parametrize('settings, side', [(TARGET, 1), (REFERENCE, 2)])
def test_2afc_incorrect_times_out(settings, side):
    plugin = make_plugin(N_response=2, **settings)
    start(plugin)
    plugin.respond(side)
    assert result(plugin) == (f'resp_{side}', 'incorrect', False)
    assert not any(a.startswith('deliver_reward') for a in plugin.h_actions)
    assert 'to_start' in plugin.h_actions
    plugin.release(side)
    assert plugin.trial_state == S.waiting_for_to


def test_2afc_no_response_is_invalid():
    plugin = make_plugin(N_response=2, **TARGET)
    start(plugin)
    plugin.fire()  # response_duration
    assert result(plugin) == ('no_response', 'invalid', False)
    assert 'to_start' not in plugin.h_actions
    assert plugin.trial_state == S.waiting_for_iti


def test_2afc_repoke_is_incorrect_without_timeout():
    plugin = make_plugin(N_response=2, **TARGET)
    start(plugin)
    plugin.h_ts += 0.1
    plugin.poke()
    assert result(plugin) == ('np', 'incorrect', False)
    assert 'to_start' not in plugin.h_actions
    assert plugin.trial_state == S.waiting_for_iti


################################################################################
# What the trial log gets
################################################################################
def test_trial_log_row():
    '''
    The values BehaviorPlugin adds to each trial's row (the context's values
    are added too). Changing this changes the trial log files.
    '''
    plugin = make_plugin(**TARGET)
    plugin.h_ts = 10.0
    plugin.poke()
    plugin.fire()  # np_duration (0.25)
    plugin.h_ts += 0.05
    plugin.withdraw()
    plugin.h_ts += 0.5
    plugin.respond(1)
    r = plugin.h_results[-1]
    expected = {
        'np_start': 10.0,
        'trial_start': 10.25,
        'np_end': pytest.approx(10.3),
        'response_start': 10.25,
        'response_ts': pytest.approx(10.8),
        'response_side': 1,
        'response': 'resp_1',
        'score': 'correct',
        'correct': True,
        'reward_delivered': True,
        'response_time': pytest.approx(0.55),
        'reaction_time': pytest.approx(0.05),
        'np_actual_duration': pytest.approx(0.3),
        'stim_start': 10.25 + plugin.context.get_value('target_delay'),
        'stim_complete': True,
        'trial_type': 'target',
        'trial_subtype': None,
        'level': 60,
    }
    for key, value in expected.items():
        assert r[key] == value, key


def test_events_for_a_trial():
    '''
    The experiment actions (events in the event log) for a hit, in order.
    '''
    plugin = make_plugin(**TARGET)
    plugin.h_actions.clear()
    start(plugin)
    plugin.respond(1)
    plugin.release(1)
    plugin.fire()  # ITI
    assert plugin.h_actions == [
        'np_start', 'np_duration_elapsed', 'trial_start',
        'response_start', 'np_end', 'resp_1_start', 'deliver_reward_1',
        'response_end', 'trial_end', 'resp_1_end', 'iti_start',
        'iti_duration_elapsed', 'iti_end', 'trial_ready',
    ]
