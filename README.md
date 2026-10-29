# Behavior paradigms 

## NAFC

The go-nogo paradigm is programmed as a special-case of the the NAFC paradigm,
so many of the parameters have a similar meaning. The code in the NAFC paradigm
does not actually define what happens for a reward and timeout. These are
defined by plugins (e.g., you can have a plugin to trigger a pellet dispenser
for a food reward, open a solenoid for a water reward, or turn a light off for
the timeout).

* **Training mode** - If checked, a reward is immediately dispensed at the
start of the response window (i.e., when the trial begins if the nose-poke hold
duration is 0). If not checked, the animal must provide a correct response to
receive the reward.

* **Manual control?** - If checked, the animal cannot trigger a trial on their
own. All trials are initiated by the user by clicking "Start Trial" in the
toolbar. If unchecked, the animal can trigger a trial by nose-poking. Even if
manual control is off, users can still initiate a trial by clicking "Start
Trial" in the toolbar.

* **Allow continuous nose-poke?** - If checked, the animal can trigger a new
trial by maintaining the current nose poke (i.e., effectively disregarding the
current trial). However, the animal must maintain the nose poke through the
intertrial interval before the new trial is triggered.

* **Repeat incorrect/invalid trials?** - If set, the stimulus from the previous
trial is presented again (logged with `trial_subtype` of `repeat`). *Early
withdraw only* repeats trials where the animal withdrew from the nose-poke
during the hold period (`early_np`). *All* repeats any trial that was not scored
as correct (note that in go/nogo, a missed target is scored as invalid and will
be repeated). *False alarm (nogo) only* (go/nogo only) repeats reference trials
that were not scored as correct. Repeats are not used during remind and warmup
trials, when a remind trial has been requested, or when a target is forced
because the maximum number of consecutive reference trials was reached.

* **Reward correct reference responses?** (N-AFC only) - If checked, a correct
response on a reference trial is rewarded. If not checked, it is scored as
correct but not rewarded (unless the trial manager's `deliver_reward` returns
True for the reference stimulus, which was the only way to reward reference
trials in earlier versions).

* **Nose-poke start duration** - Duration of nose poke required to start the
trial, in seconds. If the animal withdraws before the duration is met, no trial
begins.

* **Nose-poke hold duration** - Duration animal is required to maintain the
nose-poke after the trial begins before it can withdraw and provide a response.
If the animal withdraws before the hold duration elapses, then the trial is
scored as invalid and recorded as `early_np` in the trial log. The total time
an animal must maintain the nose-poke is the sum of nose-poke start duration
and nose-poke hold duration.

* **Response duration** - Time animal has to provide a response after the hold
period elapses. If the nose-poke hold duration is 0, then this defines the time
the animal has to respond from the start of the trial. If the animal does not
provide a response, it is recorded as a `no_response` in the trial log file.

* **Timeout duration** - Duration of timeout.

* **Intertrial interval** - Minimum duration between the end of one trial and
the start of the next trial. The end of the trial is defined as the time at
which the response period elapses, the animal provides a correct response, or
the end of the timeout (whichever is later). The start of the trial is defined
as the time at which the trial starts. This means that the animal can initiate
a nose-poke before the intertrial interval elapses and the duration the animal
has maintained the nose-poke will count towards the overal nose-poke start
duration.

The trial log includes `reward_delivered`, which indicates whether a reward
was actually dispensed on that trial.

### Toolbar controls

A trial is *in progress* from the moment the animal begins the nose-poke that
starts it until the animal responds (or the response window elapses). The next
trial is prepared (stimulus chosen) at the end of the intertrial interval.

* **Apply** - If a trial is in progress, the changes are applied as soon as
the trial ends (after its result is logged, so the trial log reflects the
settings the trial ran with). Otherwise, they are applied immediately. If the
next trial has already been prepared, it is prepared again only if the changes
affect the trial sequence or stimulus. Applying changes never shortens a
timeout or the intertrial interval.

* **Pause** - Takes effect immediately between trials (while waiting for the
animal to nose-poke). Otherwise, the experiment pauses at the end of the
intertrial interval following the current trial (after any timeout). Unchecking
Pause before then cancels the request. While paused, a trial never starts even
if the animal is nose-poking.

* **Request Remind** - While waiting for a nose-poke or during the intertrial
interval, a remind trial starts immediately. Otherwise, the next trial will be
a remind trial.

* **Start Trial** - While waiting for a nose-poke, starts the trial that has
been prepared. During the intertrial interval, skips the rest of the interval
and starts the next trial. Ignored while a trial is in progress, during a
timeout, or while the animal is at the reward port.
