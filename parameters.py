from structClass import structType
import numpy as np


def setParams():

    params = structType  # make params variable of the type struct

    # SIMULATION PARAMETERS

    params.N_SIMULATIONS = 10  # number of times to run the simulation
    params.MAX_N_STEPS = int(1e5)  # maximum number of steps to simulate
    params.MAX_N_EPISODES = 50  # maximum number of episodes to simulate (use np.inf if no max)

    # MDP PARAMETERS

    # Using greedy policy!
    params.actPolicy = 'softmax'  # Choose 'e_greedy' or 'softmax'
    params.gamma = 0.9  # discount factor
    params.alpha = 1.0  # learning rate
    params.lamb = 0  # eligibility trace parameter
    params.TLearnRate = 0.9  # learning rate for the transition matrix (0=uniform; 1=only last)
    params.softmaxInvT = 5  # soft-max inverse temperature temperature
    params.epsilon = 0.05  # probability of a random action (epsilon-greedy)
    params.PS_thresh = 0  # threshold for appending the absolute dela value in prioritized sweeping

    # let the agent explore the maze (without rewards) to learn transition model before the first episode
    params.preExplore = True
    # include a transition from goal to start in transition matrix -- this allows Need-term to wrap around
    params.Tgoal2start = True
    # When drawing reward + noise samples, truncate negative values to zero
    params.rewOnlyPositive = True
    # choose 'off-policy' (default, learns Q*) or 'on-policy' (learns Qpi) learning for updating Q-values and
    # computing gain
    params.onVSoffPolicy = 'off-policy'

    # PLANNING PARAMETERS

    # number of steps to do in planning (set to zero if no planning or to Inf to plan for as long as planning
    # beats the opportunity cost)
    params.nPlan = 20
    params.EVMthresh = 0  # minimum EVM so that planning is performed (use Inf if wish to use opportunity cost)

    # Parameters for n-step backups
    params.expandFurther = True  # Expand the last backup further
    params.planPolicy = 'softmax'  # Choose 'thompson_sampling' or 'e_greedy' or 'softmax'

    # Other planning parameters
    # boolean variable indicating if planning should happen only if the agent is at the start or goal state
    params.planOnlyAtGorS = True
    # if we want an experiment with non-stationarity wrt the reward structure (i.e. change of goal location)
    params.change_goal = True
    # if planning should occur at the previous goal state until the agent has found the new one
    params.plan_at_prev_goal = True
    params.baselineGain = 1e-10  # Gain is set to at least this value (interpreted as "information gain")
    # How to break ties on EVM (choose which sequence length is prioritized: 'min', 'max', or 'rand')
    params.tieBreak = 'min'
    # Choose 'online' or 'offline' (e.g. sleep) to determine what to use as the need-term (online: Successor
    # Representation; offline: Stationary distribution of the MDP)
    params.onlineVSoffline = 'online'
    params.remove_samestate = True  # Remove actions whose consequences lead to same state (e.g. hitting the wall)
    params.allowLoops = False  # Allow planning trajectories that return to a location appearing previously in the plan
    # Copy Q-value on planning backups (i.e., use LR=1.0 for updating Q-values in planning and for computing gain)
    params.copyQinPlanBkps = False
    params.setAllGainToOne = False  # Set the gain term of all items to one (for debugging purposes)
    params.setAllNeedToOne = False  # Set the need term of all items to one (for debugging purposes)
    # Set the need term of all items to zero, except for the current state (for debugging purposes)
    params.setAllNeedToZero = False

    # PLOTTING SETTINGS

    params.plot_agent = False
    params.plot_Q = False

    return params
