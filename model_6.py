"""Class that represents the interaction network of Stack overflow."""

# Imports
import scipy.stats
import numpy as np
import user_6 as agent

class system:
    """
    Framework of the model that represents the interaction network of Stack overflow.

    Attributes
    ----------
    new_users : int
        number of users added every timestep
    distr : dict
        contains the type of distributions from which the probabilities are sampled
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
    draw_normal(mu, sigma)
        Draw probability from normal distribution.
    draw_uniform()
        Draw probability from uniform distribution.
    draw_exponential(alpha)
        Draw probability from exponential distribution.
    calc_cdf(pdf)
        Calculate the cummulative distribution function of a given pdf.
    """

    def __init__(self, n, tags, distr_ask='normal', distr_answer='normal', distr_interact='normal', distr_active='normal'):
        """
        Initialization of a interaction network.

        Parameters
        ----------
        n : int
            number of users added every timestep
        tags : str
            .txt file containing probabilities of the different communities
        distr_ask : str
            distribution of the probability to ask a question, default is normally distributed
        distr_answer : str
            distribution of the probability to answer a question, default is normally distributed
        distr_interact : str
            distribution of the probability to upvote a question/answer, default is normally distributed
        distr_active : str
            distribution of the probability to be active, default is normally distributed
        """
        self.new_users = n

        # Distributions for the interaction parameters of the users
        self.distr = {'p_ask': distr_ask, 'p_answer': distr_answer, 'p_interact': distr_interact, 'p_active': distr_active}

        # Calculate the cummulative distribution of the tags
        tag_pdf = np.loadtxt(tags, usecols=1)
        tag_pdf = tag_pdf / np.sum(tag_pdf)
        self.tag_cdf = self.calc_cdf(tag_pdf)

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
        for i, param in enumerate(self.distr):
            if self.distr[param] == 'normal':
                p = self.draw_normal()
            elif self.distr[param] == 'uniform':
                p = self.draw_uniform()
            elif self.distr[param] == 'exponential':
                p = self.draw_exponential(3)
            else:
                raise ValueError('Unkown distribution.')
            setattr(user, param, p)
            setattr(user, param + '_begin', p)
            
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
        
    def draw_normal(self, mu=0.5, sigma=0.25):
        """
        Draw probability from normal distribution.

        Parameters
        ----------
        mu : float
            mean of the distribution
        sigma : float
            std of the distribution
        
        Returns
        -------
        p : float
            probability
        """
        # value bounded between 0 and 1
        lower = 0
        upper = 1

        return scipy.stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma)
    
    def draw_uniform(self):
        """
        Draw probability from uniform distribution.

        Returns
        -------
        p : float
            probability
        """
        return np.random.uniform()
    
    def draw_exponential(self, alpha):
        """
        Draw probability from exponential distribution.

        Parameters
        ----------
        alpha : int
            rate
        
        Returns
        -------
        p : float
            probability
        """
        return scipy.stats.truncexpon.rvs(alpha)/alpha

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


if __name__ == '__main__':

    n = 20
    increase = 20
    test = system(n, 'tags.txt', distr_ask='exponential', distr_answer='exponential', distr_interact='exponential', distr_active='exponential')
    
    for tag, users in enumerate(test.tags):
        ids = []
        for user in users:
            ids.append(user)
        print(tag, ids)
    
    test.run(10)
    print('finished')
