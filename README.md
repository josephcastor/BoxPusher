# BoxPusher

## Running the solver:
Belt of size 5:
`python3 mcts.py`

Belt of size 9:
`python3 mcts_9.py`

Belt of size 13:
`python3 mcts_13.py`

### Further specifications:
`-r` to use random simulation rather than heuristic based simulation in the MCTS (off by default)

`-t [planning_time]` to specify planning time for each action (3 by default)

`-s [steps]` to specify number of time steps for each trial (30 by default)

For example:

`python3 mcts_9.py -r -t 1 -s 20`

Will run the solver with random simulation, with 1 second of planning time and 20 steps per trial.


### Test Suites
To run the test suites, simply run the files in the `problems` directory (no arguments needed).

For example:
`python3 problems/greedy_actions.py`
