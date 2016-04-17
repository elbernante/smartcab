import os
import shutil
import csv
from itertools import product
from smartcab.agent import LearningAgent
from smartcab.environment import Environment
from smartcab.simulator import Simulator

LOG_BOOK = 'log_book/'
Q_LOG = 'q_log/'

def write_to_csv(filename, header, data):
    with open(filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def prepare_folder(*paths):
    for p in paths:
        if os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p)


class Observer(object):

    def __init__(self, filename, alpha, gamma, sigma):
        self.filename = filename
        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma

        self.deadline = 0
        self.distance = 0
        self.steps = 0
        self.rewards = 0
        self.penalties = 0

        self.trials = []    # CSV format is (deadline, distance, steps, rewards, penalties)


    def notify(self, source, event_name, *args, **kwargs):
        
        if event_name == "step":
            assert isinstance(source, LearningAgent), "Source should be a LearningAgent!"
            reward = kwargs['reward']
            self.steps += 1
            self.rewards += reward
            self.penalties += reward if reward < 0 else 0

        elif event_name == 'reset':
            assert isinstance(source, Environment), "Source should be Environment!"
            agent_state = source.agent_states[source.primary_agent]
            self.deadline = agent_state['deadline']
            self.distance = source.compute_dist(agent_state['location'], agent_state['destination'])

            self.steps = 0
            self.rewards = 0
            self.penalties = 0

        elif event_name == 'end_trial':
            assert isinstance(source, Environment), "Source should be Environment!"
            self.trials.append([self.deadline,
                                self.distance,
                                self.steps,
                                self.rewards,
                                self.penalties])

    def save_q_table(self, agent):
        Q = agent.Q_
        s_labels, _ = zip(*Q.states_)

        header = [str(s_labels)] + [str(a) for a in Q.actions_] + ['E(s)', 'Hit(s)']
        
        a_e = lambda s: [str(s)] + [Q[(s, a)] for a in Q.actions_] + [Q.E[s], Q.hit[s]]
        grid = [a_e(s) for s in Q.states()]
        
        write_to_csv(Q_LOG + self.filename, header, grid)


    def wrap_up(self):
        write_to_csv(LOG_BOOK + self.filename,
            ["deadline", "distance", "steps", "rewards", "penalties"],
            self.trials)


def run():
    alpha = [0.01, 0.1, 0.5, 0.9, 1.0]
    gamma = [0.01, 0.1, 0.3, 0.9, 1.0]
    sigma = [0.01, 0.1, 0.5, 0.8, 1.0]
    
    params = product(alpha, gamma, sigma)
    
    indexer = ['filename', 'alpha', 'gamma', 'sigma']
    indices = []

    prepare_folder(LOG_BOOK, Q_LOG)

    for i, p in enumerate(params):
        indices.append(["{}.csv".format(i)] + list(p))

        o = Observer(*indices[-1])
        e = Environment()

        a = e.create_agent(LearningAgent, *p)
        setattr(a, "observer", o)

        e.set_primary_agent(a, enforce_deadline=True) 
        sim = Simulator(e, update_delay=0.0, display=False)
        setattr(sim, "observer", o)

        sim.run(n_trials=100)

        o.save_q_table(a)
        o.wrap_up()

    write_to_csv("log_index.csv", indexer, indices)


if __name__ == '__main__':
    run()
