# pend_magman_wrappers
Wrappers for an inverted pendulum and magnetic manipulator to be used in the baseline algorithms alongside OpenAI gym environments

Download or clone the wrappers to a folder and install them (probably it would be good to pre-install gym with pip install gym)

Run 
> pip install -e . 

in the gym-tim folder from the terminal.

call it by using 

> import gym_tim

> gym.make('tim_magman-v0')


Pre-requisite libraries for the magman and pendulum can be found here:

https://github.com/timdebruin/CoR-control-benchmarks


# Commands

The wrappers include all the basic commands that are part of the gym library such as 

> env.reset()

> env.step()

> env.render()

In addition you can use these commands for soft-resets and evaluation of the environment

> env.reset_model_to_specific_state(state)

> env.get_all_states()

The wrappers include choice for the input between the standard position and velocity or to output a basis-function vector of 441 parameters (grid of 21X21). Those can be activated from

> self.basis_fun_states = True

The hard reset env.reset() can also be modified to reset the pendulum and magman within the whole domain or reset them to a specific position (for pendulum that position is the low equilibrium [o,o] and for magman is left of the first coil at [0,0]).

For magman

> self.reset_to_random=True

For pendulum

> self.reset_to_0 = False






 
