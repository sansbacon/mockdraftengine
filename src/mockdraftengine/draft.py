"""
draft.py 

library for simulating draft
is NFL-only right now

basic idea is to use ADP data to simulate draft,
subject to round and overall positional constraints

Example:

n_teams = 12
n_rounds = 15
df = pd.read_csv('adp.csv', index_col='player') 

pos_min = {
 1: {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0},
 2: {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0},
 3: {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0},
 4: {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0},
 5: {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0},
 6: {'QB': 0, 'RB': 0, 'WR': 1, 'TE': 0},
 7: {'QB': 0, 'RB': 1, 'WR': 1, 'TE': 0},
 8: {'QB': 0, 'RB': 2, 'WR': 2, 'TE': 0},
 9: {'QB': 0, 'RB': 3, 'WR': 3, 'TE': 0},
 10: {'QB': 0, 'RB': 3, 'WR': 4, 'TE': 0},
 11: {'QB': 0, 'RB': 4, 'WR': 4, 'TE': 0},
 12: {'QB': 0, 'RB': 4, 'WR': 5, 'TE': 0},
 13: {'QB': 0, 'RB': 4, 'WR': 5, 'TE': 1},
 14: {'QB': 0, 'RB': 4, 'WR': 5, 'TE': 1},
 15: {'QB': 1, 'RB': 4, 'WR': 5, 'TE': 1}
}

pos_max = {
 1: {'QB': 0, 'RB': 1, 'WR': 1, 'TE': 1},
 2: {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 2},
 3: {'QB': 1, 'RB': 3, 'WR': 3, 'TE': 2},
 4: {'QB': 1, 'RB': 4, 'WR': 4, 'TE': 2},
 5: {'QB': 2, 'RB': 4, 'WR': 5, 'TE': 3},
 6: {'QB': 2, 'RB': 4, 'WR': 5, 'TE': 3},
 7: {'QB': 2, 'RB': 5, 'WR': 6, 'TE': 3},
 8: {'QB': 3, 'RB': 5, 'WR': 6, 'TE': 3},
 9: {'QB': 3, 'RB': 6, 'WR': 7, 'TE': 3},
 10: {'QB': 3, 'RB': 6, 'WR': 7, 'TE': 3},
 11: {'QB': 3, 'RB': 6, 'WR': 7, 'TE': 3},
 12: {'QB': 3, 'RB': 7, 'WR': 8, 'TE': 3},
 13: {'QB': 3, 'RB': 7, 'WR': 8, 'TE': 3},
 14: {'QB': 3, 'RB': 7, 'WR': 8, 'TE': 3},
 15: {'QB': 3, 'RB': 7, 'WR': 8, 'TE': 3}
}

def convert_pick_number(p, teams=12):
    '''Converts 1.01 to 1, 2.01 to 13, etc.'''
    # handles 1.01, etc.
    try:
        draft_round, pick = [int(s) for s in p.split('.0')]

    # handles 1.10 which appears as simply 1.1
    except ValueError:
        draft_round, pick = [int(s) for s in p.split('.')]
        pick = 10 if pick == 1 else pick
    return ((draft_round - 1) * teams) + pick

dtypes = {
 'player': str,
 'pos': str,
 'team': str,
 'bye': int,
 'adp': float,
 'sdev': float,
 'minpk': str,
 'maxpk': str,
 'n': int
}
 
df = pd.read_csv('adp.csv', dtype=dtypes, index_col='player')

# fix min/max columns
df.minpk = df.minpk.apply(lambda x: convert_pick_number(x, n_teams))
df.maxpk = df.maxpk.apply(lambda x: convert_pick_number(x, n_teams))

"""
from math import ceil
import logging

import pandas as pd
from scipy.stats import truncnorm


class PlayerData:
    def __init__(self, fn=None):
        """Object instantiation

        Args:
            fn (str): data source filename
        """
        logging.getLogger(__name__).addHandler(logging.NullHandler())
        self.players = None

    def load_from_csv(self, fn=None):
        """Loads data from file"""
        if fn:
            self.players = pd.read_csv(fn)
        else:
            self.players = pd.read_csv(self.fn)
        return self.players


class RoundConstraint:
    def __init__(self):
        """Object instantiation"""
        logging.getLogger(__name__).addHandler(logging.NullHandler())

    def add_constraint(self, constraint, constraint_type):
        """Adds min or max constraint
        
        Args:
            constraint (dict): key of round, value of dict
            constraint_type (str): 'min' or 'max'

        Returns:
            None
        """
        if constraint_type == 'min':
            self.min_constraint = constraint
        elif constraint_type == 'max':
            self.max_constraint = constraint
        else:
            raise ValueError('invalid constraint type')


class Draft:
    """Simulates fantasy draft"""
    def __init__(self,
                 player_data,
                 constraint, 
                 n_teams, 
                 n_rounds
                 ):
        """Object instantiation

        Args:
            player_data (PlayerData): valid Playerdata object
            constraint (RoundConstraint): valid RoundConstraint object
            n_teams (int): the number of teams in league
            n_rounds (int): the number of rounds in the draft
        """
        logging.getLogger(__name__).addHandler(logging.NullHandler())
        self.player_data = player_data
        self.constraint = constraint
        self.n_teams = n_teams
        self.n_rounds = n_rounds
        
    @property
    def n_picked(self):
        return self.n_teams * self.n_rounds
        
    def sim(self):
        """Simulates fantasy draft"""
        df = self.player_data
        skipped = {i: [] for i in range(1, self.n_picked)}
        pick = 1
        picked = []

        while len(picked) < self.n_picked:
            outer_bound = ceil(pick / self.n_rounds)
            if outer_bound < 3:
                outer_bound = 3
            elif outer_bound > 50:
                outer_bound = 50
                
            tmp = df.loc[~df.index.isin(picked)].copy()
            tmp = tmp.iloc[0:outer_bound].sample(outer_bound)
            for row in tmp.itertuples():
                mu = row.adp
                sigma = row.sdev
                lower = row.minpk
                upper = row.maxpk
                a = (lower - mu) / sigma
                b = (upper - mu) / sigma
                try:
                    X = truncnorm(a, b, loc=mu, scale=sigma)
                    sample = X.rvs(1)[0]
                    if sample < pick + 1:
                        print(f'Selected {row.Index} (adp {mu}) with pick {pick} - sample was {sample}')
                        picked.append(row.Index)
                        pick += 1
                        break
                except ValueError as e:
                    pass
                skipped[pick].append(row.Index)

        return picked, skipped
    
