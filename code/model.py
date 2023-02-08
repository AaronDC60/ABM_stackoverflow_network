"""Class that represents the interaction network of Stack overflow."""

# Imports
import numpy as np
from sklearn.linear_model import LinearRegression

import user as agent
import utils


class system:
    """
    Framework of the model that represents the interaction network of Stack overflow.

    Attributes
    ----------
    new_users : int
        number of users added every timestep
    upvote_treshold : int
        minimum reputation to gain upvoting privilige
    upvote_bias : int
        number of upvotes a user is satisfied with
    distr : list
        contains the type and parameters of the distributions from which the probabilities are sampled
    tag_cdf : numpy.ndarray
        cummulative distribution function of the tags (communities)
    tags : list
        contains the ids of the users with a certain tag for all tags
    users : list
        contains all the users in the system
    questions : list
        all the questions ever asked during the simulation

    Methods
    -------
    determine_tag()
        Determine the tag of a user.
    create_user(i)
        Create a new user.
    step()
        Single timestep of the model.
    run(t)
        Execute the model for a certain number of timesteps.
    """

    def __init__(self, n, tags, treshold=15, bias=12, distr=[[0.5, 0.25], [0.5, 0.25], [0.5, 0.25], [0.5, 0.25]]):
        """
        Initialize an interaction network.

        Parameters
        ----------
        n : int
            number of users added every timestep
        tags : str
            .txt file containing probabilities of the different communities
        treshold : int
            minimum reputation to gain upvoting privilige, default is 15
        bias : int
            number of upvotes a user is satisfied with, default is 12
        distr : list (4 x 2)
            contains the mean and std (as a list) of the distributions from which the probabilities are sampled
            the first list is for p_ask followed by p_answer, p_interact and p_active
            default values are mean 0.5 and std 0.25 (normal distribution)
            for uniform distribution, set the mean to None
            for exponential distribution set the mean equal to the rate and the std to None
        """
        self.new_users = n
        self.upvote_treshold = treshold
        self.upvote_bias = bias

        # Distributions for the interaction parameters of the users
        self.distr = distr

        # Calculate the cummulative distribution of the tags
        tag_pdf = np.loadtxt(tags, usecols=1)
        tag_pdf = tag_pdf / np.sum(tag_pdf)
        self.tag_cdf = utils.calc_cdf(tag_pdf)

        self.tags = [[] for _ in range(len(tag_pdf))]
        self.users = []
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

    def create_user(self, i):
        """
        Create a new user.

        Parameters
        ----------
        i : int
            id of the user

        Returns
        -------
        user : .user
            new user
        """
        # Tag
        tag = self.determine_tag()
        self.tags[tag].append(i)

        # User
        user = agent.user(self, i, tag)

        # Probabilities
        attributes = ['p_ask', 'p_answer', 'p_interact', 'p_active']
        for i, param in enumerate(self.distr):
            if param[0] is None:
                # Uniform distribution
                p = utils.draw_uniform()
            elif param[1] is None:
                # Exponential distribution
                p = utils.draw_exponential(param[0])
            else:
                # Normal distribution
                p = utils.draw_normal(param[0], param[1])

            setattr(user, attributes[i], p)
            setattr(user, attributes[i] + '_begin', p)

        return user

    def step(self):
        """Single timestep of the model."""
        # Add new users to the system
        for _ in range(self.new_users):
            user = self.create_user(len(self.users))
            self.users.append(user)

        # Iterate over users based on activity, most active users go first
        order = list(np.copy(self.users))
        order.sort(key=lambda x: x.p_active, reverse=True)
        for user in order:
            user.step()

    def run(self, t):
        """
        Execute the model for a certain number of timesteps.

        Parameters
        ----------
        t : int
            number of timesteps
        """
        for _ in range(t):
            self.step()

    def reset(self):
        """Reset the system (does not change the parameter settings)."""
        self.tags = [[] for _ in range(len(self.tag_cdf))]
        self.users = []
        self.questions = []

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
