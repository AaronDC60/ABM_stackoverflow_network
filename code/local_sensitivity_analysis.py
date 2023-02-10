"""Local sensitivity analysis (OFAT)."""

import numpy as np
import multiprocessing as mp
import pandas as pd

from model import *

variables = {
    'num_vars' : 6,
    'names': ['treshold', 'bias', 'mu_p_ask', 'mu_p_answer', 'mu_p_upvote', 'mu_p_active'],
    'bounds': [[0, 40], [0, 40], [0, 1], [0, 1], [0, 1], [0, 1]]}

def simulation(var, i):

    runs = 10
    n_samples = 15

    # Dictionary to store the results
    data_upvotes = {var: []}
    data_reputation = {var: []}
    data_questions = {var: []}
    data_answers = {var: []}

    if var == 'treshold' or var == 'bias':
        values = np.linspace(*variables['bounds'][i], num=n_samples, dtype=int)
    else:
        values = np.linspace(*variables['bounds'][i], num=n_samples)

    for ind, value in enumerate(values):
        # Initialize the default model
        stackoverflow = network(250, 'tags.txt')

        # Change the right parameter
        if var == 'treshold':
            stackoverflow.upvote_treshold = value
        elif var == 'bias':
            stackoverflow.upvote_bias = value
        elif var == 'mu_p_ask':
            stackoverflow.distr[0][0] = value
        elif var == 'mu_p_answer':
            stackoverflow.distr[1][0] = value
        elif var == 'mu_p_upvote':
            stackoverflow.distr[2][0] = value
        elif var == 'mu_p_active':
            stackoverflow.distr[3][0] = value
        elif var == 'std_p_ask':
            stackoverflow.distr[0][1] = value
        elif var == 'std_p_answer':
            stackoverflow.distr[1][1] = value
        elif var == 'std_p_upvote':
            stackoverflow.distr[2][1] = value
        elif var == 'std_p_active':
            stackoverflow.distr[3][1] = value
        else:
            ValueError('Unknown parameter given (%s)' %var)
        
        data_votes = []
        data_rep = []
        data_q = []
        data_a = []

        for _ in range(runs):
            # Run the simulation for 20 timesteps
            stackoverflow.run(20)
            # Collect output
            data_votes.append(stackoverflow.get_regression_coeff(data='upvotes', binsize=5))
            data_rep.append(stackoverflow.get_regression_coeff(data='reputation', binsize=125))
            data_q.append(len(stackoverflow.questions))
            n_answers = 0
            for question in stackoverflow.questions:
                n_answers += len(question.answers)
            data_a.append(n_answers)
            # Reset the network
            stackoverflow.reset()
        
        print((ind + 1)/n_samples, var)

        data_upvotes[var].append(data_votes)
        data_reputation[var].append(data_rep)
        data_questions[var].append(data_q)
        data_answers[var].append(data_a)
    
    df_upvotes = pd.DataFrame.from_dict(data_upvotes)
    df_reputation = pd.DataFrame.from_dict(data_reputation)
    df_questions = pd.DataFrame.from_dict(data_questions)
    df_answers = pd.DataFrame.from_dict(data_answers)

    df_upvotes.to_csv('ofat_upvotes_%s.csv'%var)
    df_reputation.to_csv('ofat_reputation_%s.csv'%var)
    df_questions.to_csv('ofat_questions_%s.csv'%var)
    df_answers.to_csv('ofat_answers_%s.csv'%var)

def merge(output):
    # Group some of the dataframes in the same file
    df1 = pd.read_csv('ofat_%s_bias.csv'%output, sep=',', usecols=['bias'])
    df2 = pd.read_csv('ofat_%s_treshold.csv'%output, sep=',', usecols=['treshold'])
    df3 = pd.read_csv('ofat_%s_mu_p_ask.csv'%output, sep=',', usecols=['mu_p_ask'])
    df4 = pd.read_csv('ofat_%s_mu_p_answer.csv'%output, sep=',', usecols=['mu_p_answer'])
    df5 = pd.read_csv('ofat_%s_mu_p_upvote.csv'%output, sep=',', usecols=['mu_p_upvote'])
    df6 = pd.read_csv('ofat_%s_mu_p_active.csv'%output, sep=',', usecols=['mu_p_active'])

    df1['treshold'] = df2['treshold']
    df1['mu_p_ask'] = df3['mu_p_ask']
    df1['mu_p_answer'] = df4['mu_p_answer']
    df1['mu_p_upvote'] = df5['mu_p_upvote']
    df1['mu_p_active'] = df6['mu_p_active']

    df1.to_csv('ofat_%s.csv'%output)
    

if __name__ == '__main__':
    processes = []
    for i, var in enumerate(variables['names']):
        p = mp.Process(target=simulation, args=(var, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    # One csv file per output parameter
    output_param = ['answers', 'questions', 'reputation', 'upvotes']

    for param in output_param:
        merge(param)
