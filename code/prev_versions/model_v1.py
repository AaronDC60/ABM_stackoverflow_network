"""Class that represents the interaction network of Stack overflow."""

# Imports
import numpy as np
from sklearn.linear_model import LinearRegression

import user_v1 as agent

class system:

    def __init__(self, n, tags):
        # n is the total number of agents
        # tags is a txt file with the probabilities of the different topics

        # Calculate the cummulative distribution of the tags
        tag_pdf = np.loadtxt(tags, usecols=1)
        tag_pdf = tag_pdf / np.sum(tag_pdf)

        self.tag_cdf = self.calc_cdf(tag_pdf)

        # List with all the users per tag
        self.tags = [[] for _ in range(len(tag_pdf))]

        self.users = []
        # Generate all the user nodes
        for i in range(n):
            self.users.append(self.create_user(self, i))

        self.questions = []
        
    def determine_tag(self):
        """
        Determine the tag of a user.

        Returns
        -------
        tag : int
            tag of the user
        """
        tag = 0
        u = np.random.uniform()
        while u > self.tag_cdf[tag]:
            tag += 1

        return tag
    
    def calc_pdf(self, array):
        """
        Calculate the probability density function of an array.

        Parameters
        ----------
        array : numpy.ndarray
            list with outcomes
        
        Returns
        -------
        pdf : numpy.ndarray
            probability density function
        """
        pdf = np.zeros(np.max(array) + 1)
        for element in array:
            pdf[element] += 1
        
        return pdf / np.sum(pdf)
    
    def calc_cdf(self, pdf):
        """
        Calculate the cummulative distribution function of a given pdf.

        Parameters
        ----------
        pdf : numpy.ndarray
            probability density function

        Returns
        -------
        cdf : numpy.ndarray
            cummulative distribution function
        """
        cdf = []
        value = 0

        for p in pdf:
            value += p
            cdf.append(value)

        return np.array(cdf)

    def create_user(self, model, i):
        tag = self.determine_tag()
        self.tags[tag].append(i)

        return agent.user(model, i, tag)

    def step(self, time):
        
        for user in self.users:
            user.step()

        if time < 10:
            # Add 200 new users to the system
            for _ in range(200):
                user = self.create_user(self, len(self.users))
                # Starting reputation of 1
                user.reputation = 1
                self.users.append(user)

    def run(self, t):

        # t is the number of timesteps
        for i in range(t):
            self.step(i)

    def get_upvote_distr(self, binsize):
        """
        Get the distribution of upvotes given per user.

        Parameters
        ----------
        binsize : float
            length of one interval

        Returns
        -------
        pdf : numpy.ndarray
            probability density function of the number of upvotes
        bins : numpy.ndarray
            edges of the bins
        """
        # Get the data on the upvotes
        upvotes = []
        for user in self.users:
            upvotes.append(user.n_questions_upvoted + user.n_answers_upvoted)

        bins = np.arange(binsize, max(upvotes) + binsize + 1, binsize)
        pdf = np.zeros(len(bins))
        for value in upvotes:
            pdf[(value // binsize)] += 1

        pdf /= np.sum(pdf)

        return pdf, bins

    def get_reputation_distr(self, binsize):
        """
        Get the distribution of reputation.

        Parameters
        ----------
        binsize : float
            length of one interval

        Returns
        -------
        pdf : numpy.ndarray
            probability density function of the reputation
        bins : numpy.ndarray
            edges of the bins
        """
        # Get the data on reputation
        reputation = []
        for user in self.users:
            reputation.append(user.reputation)

        bins = np.arange(binsize, max(reputation) + binsize + 1, binsize)
        pdf = np.zeros(len(bins))
        for value in reputation:
            pdf[(value // binsize)] += 1

        pdf /= np.sum(pdf)

        return pdf, bins

    def get_regression_coeff(self, data='upvotes', binsize=20):
        """
        Calculate the linear regression coefficient of the distribution of upvotes or reputation.

        Parameters
        ----------
        data : str ('upvotes' or 'reputation')
            specifies for which distribution the coefficient should be calculated, default is upvotes
        binsize : float
            length of the interval used in calculating the pdf

        Returns
        -------
        coeff : float
            coefficient of the linear regression line
        """
        if data == 'upvotes':
            pdf, bins = self.get_upvote_distr(binsize)
        else:
            pdf, bins = self.get_reputation_distr(binsize)

        # Calculate the log of the data
        pdf_log = []
        bins_log = []

        for ind, value in enumerate(pdf):
            if value != 0:
                bins_log.append(np.log10(bins[ind]))
                pdf_log.append(np.log10(value))

        lin_model = LinearRegression().fit(np.array(bins_log).reshape((-1, 1)), pdf_log)

        return (lin_model.coef_)[0]
