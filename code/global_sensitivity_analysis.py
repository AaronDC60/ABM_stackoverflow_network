"""Global sensitivity analysis."""

# Imports
import multiprocessing as mp
import pandas as pd
from SALib.sample import saltelli

from model import *

variables = {
    'num_vars' : 5,
    'names': ['bias', 'mu_p_upvote', 'mu_p_active','std_p_ask', 'std_p_answer'],
    'bounds': [[1, 40], [0, 1], [0, 1], [0.01, 1], [0.01, 1]]}

runs = 8

# Calculate sample points
n_samples = 512
param_values = saltelli.sample(variables, n_samples, calc_second_order=False)

def analysis(run):
    data = pd.DataFrame(index=range(run * len(param_values), run * len(param_values) + len(param_values)), columns=['bias', 'mu_p_upvote', 'mu_p_active', 'std_p_ask', 'std_p_answer'])
    data['coeff_upvotes'], data['coeff_reputation'], data['n_questions'], data['n_answers'] = None, None, None, None

    for ind, setting in enumerate(param_values):
        # Setup the model
        stackoverflow = network(150, 'tags.txt', bias=int(setting[0]))
        # Change settings
        stackoverflow.distr[2][0] = setting[1]
        stackoverflow.distr[3][0] = setting[2]
        stackoverflow.distr[0][1] = setting[3]
        stackoverflow.distr[1][1] = setting[4]
        
        stackoverflow.run(20)

        # Determine the total number of answers
        n_answers = 0
        for question in stackoverflow.questions:
            n_answers += len(question.answers)

        # Write results and setting to dataframe
        data.at[run * len(param_values) + ind, 'bias'] = setting[0]
        data.at[run * len(param_values) + ind, 'mu_p_upvote'] = setting[1]
        data.at[run * len(param_values) + ind, 'mu_p_active'] = setting[2]
        data.at[run * len(param_values) + ind, 'std_p_ask'] = setting[3]
        data.at[run * len(param_values) + ind, 'std_p_answer'] = setting[4]

        data.at[run * len(param_values) + ind, 'coeff_upvotes'] = stackoverflow.get_regression_coeff(data='upvotes', binsize=5)
        data.at[run * len(param_values) + ind, 'coeff_reputation'] = stackoverflow.get_regression_coeff(data='reputation', binsize=125)
        data.at[run * len(param_values) + ind, 'n_questions'] = len(stackoverflow.questions)
        data.at[run * len(param_values) + ind, 'n_answers'] = n_answers

        print(run, (ind+1) / len(param_values))
    
    # Write to csv
    data.to_csv('global_sa_%s.csv'%run)

if __name__ == '__main__':
    processes = []
    for i in range(runs):
        p = mp.Process(target=analysis, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
