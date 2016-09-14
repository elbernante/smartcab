import random
from itertools import product
from math import exp
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class Ddict(dict):
    """Dictionary that returns a default value if key doesn't exists"""
    def __init__(self, default=0.0):
        super(Ddict, self).__init__()
        self.default_value = default

    def __getitem__(self, key):
        return dict.get(self, key, self.default_value)

class QTable(Ddict):
    def __init__(self, states=[], actions=[]):
        super(QTable, self).__init__(default=0.0)

        self.states_ = states
        self.actions_ = actions
        self.E = Ddict(default=1.0)     # Epsilon(s):
                                        #   1.0 - choose random(action)
                                        #   0.0 - argmax_a Q(state, action)
        self.hit = Ddict(default=0)     # Hit count for visited states

    def states(self):
        """Returns iterator for all possible states"""
        _, s_values = zip(*self.states_)
        return product(*s_values)

    def __str__(self):
        """Pretty print Q table with epsilon for each state"""
        s_labels, _ = zip(*self.states_)
        epsilon = "E(s)"
        hit_count = "Hit(s)"

        # Max width for each column
        max_l_s = max(len(str(s_labels)), *[len(str(s)) for s in self.states()])
        max_l_v = max([len(str(a)) for a in self.actions_] +
                      [len(str(v)) for v in self.itervalues()])
        max_l_e = max([len(epsilon)] + [len(str(e)) for e in self.E.itervalues()])
        max_l_h = max([len(hit_count)] + [len(str(h)) for h in self.hit.itervalues()])

        template = ("{:%d} |" % max_l_s) + \
                   ((" {:>%d} " % max_l_v) * len(self.actions_)) + \
                   ("   | {:>%d} " % max_l_e) + \
                   ("|  {:>%d}" % max_l_h)

        header = template.format(s_labels, *(self.actions_ + [epsilon, hit_count]))
        header = "\n".join(["-" * len(header), header, "-" * len(header)]) + "\n"
        a_e = lambda s: [self[(s, a)] for a in self.actions_] + [self.E[s], self.hit[s]]
        grid = [template.format(s, *a_e(s)) for s in self.states()]

        return header + "\n".join(grid)


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    LIGHT = ['green', 'red']
    ACTIONS = [None, 'forward', 'left', 'right']
    INTERSECTION = ['green_busy', 'green_clear', 'red_busy', 'red_clear']

    def __init__(self, env, alpha=0.9, gamma=0.05, sigma=.8):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        self.Q_ = QTable(zip(['Intersection', 'Waypoint'],
                             [LearningAgent.INTERSECTION, LearningAgent.ACTIONS]),
                         LearningAgent.ACTIONS)

        self.alpha_ = alpha                             # Learning rate
        self.gamma_ = gamma                             # Discount factor
        self.delta_ = 1.0 / len(LearningAgent.ACTIONS)  # Action influence
        self.sigma_ = sigma                             # Inverse sensitivity:
                                                        #       low value = sensitive to small Q value change; more exploration
                                                        #       high value = sensitive only to large Q value change; less exploration

        self._actions = LearningAgent.ACTIONS[:]
        self._prev_state = None
        self._prev_action = None
        self._prev_reward = 0.0


    def reset(self, destination=None):
        self.planner.route_to(destination)

        self._prev_state = None
        self._prev_action = None
        self._prev_reward = 0.0


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        self.state = self._read_state(inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)

        action = self._pi(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        if self._prev_state is not None:
            self._update_q(self._prev_state, self._prev_action, self._prev_reward, self.state)
        self._prev_state = self.state
        self._prev_action = action
        self._prev_reward = reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

        # For metrics gathering
        if getattr(self, "observer", None):
            self.observer.notify(self, 'step', reward=reward)
        

    def _read_state(self, light, oncoming, left, waypoint):
        assert light in LearningAgent.LIGHT, "Invalid light!"
        assert oncoming in LearningAgent.ACTIONS, "Invalid action!"
        assert left in LearningAgent.ACTIONS, "Invalid action!"
        assert waypoint in LearningAgent.ACTIONS, "Invalid action!"

        # Merge similar states
        xsection = None
        if light == 'green':
            if oncoming == 'forward':
                xsection = 'green_busy'
            else:
                xsection = 'green_clear'
        elif light == 'red':
            if oncoming == 'left' or left == 'forward':
                xsection = 'red_busy'
            else:
                xsection = 'red_clear'

        assert xsection in LearningAgent.INTERSECTION, "Invalid intersection!"

        return (xsection, waypoint)

    def _pi(self, state):
        random.shuffle(self._actions)   # shuffle actions to randomize selection in cases of equal values
        keys = [(state, a) for a in self._actions]

        max_a = lambda: max(keys, key=lambda k: self.Q_[k])[1]  # Greedy selection
        rand_a = lambda: random.choice(self._actions)           # Random selection

        # Apply greedy selection with probabity: 1 - epsilon[state], else random
        idx = 0 if random.random() <= 1. - self.Q_.E[state] else 1
        action = (max_a, rand_a)[idx]()
        
        print "\nState: {} | Selection: {} | Action: {}".format(state, ('argmax_a Q(s,a)', 'random(a)')[idx], action) # [debug]

        return action


    def _update_q(self, state, action, reward, state_prime):
        util_prime = max([self.Q_[(state_prime, a)] for a in self._actions])
        util_sate = reward + self.gamma_ * util_prime

        # Update Q[s,a]
        q_sa =  self.Q_[(state, action)]
        self.Q_[(state, action)] = (1.0 - self.alpha_) * q_sa + self.alpha_ * util_sate

        # Update E[s]
        e = exp(-abs(self.Q_[(state, action)] - q_sa) / self.sigma_)
        f_sad = (1. - e) / (1. + e)
        self.Q_.E[state] = (1. - self.delta_) * self.Q_.E[state] + self.delta_ * f_sad

        # Icrement state hit count
        self.Q_.hit[state] += 1 


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print a.Q_      # Show Q-table with epsilon for each state     # [debug]   


if __name__ == '__main__':
    run()
