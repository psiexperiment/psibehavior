# Behavior paradigms 

## NAFC

The go-nogo paradigm is programmed as a special-case of the the NAFC paradigm,
so many of the parameters have a similar meaning. The code in the NAFC paradigm
does not actually define what happens for a reward and timeout. These are
defined by plugins (e.g., you can have a plugin to trigger a pellet dispenser
for a food reward, open a solenoid for a water reward, or turn a light off for
the timeout).

* **Training mode** - If checked, a reward is immediately dispensed when
a trial begins. If not checked, the animal must provide a correct response to
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

* **Repeat incorrect/invalid trials?** - TODO

* **Nose-poke start duration** - Duration of nose poke required to start the
trial, in seconds. If the animal withdraws before the duration is met, no trial
begins.

* **Nose-poke hold duration** - Duration animal is required to maintain the
nose-poke after the trial begins before it can withdraw and provide a response.
If the animal withdraws before the hold duration elapses, then the trial is
scored as incorrect and recorded as `early_np` in the trial log. The total time
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
