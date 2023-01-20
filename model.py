"""Class that represents the interaction network of Stack overflow."""

# Imports
import numpy as np
from user import *

class system:

    def __init__(self, n, tags):
        # n is the total number of agents
        # tags is a txt file with the probabilities of the different topics

        # Calculate the cummulative distribution of the tags
        tag_pdf = np.loadtxt(tags, usecols=1)
        tag_pdf = tag_pdf / np.sum(tag_pdf)

        tag_cdf = self.calc_cdf(tag_pdf)

        # List with all the users per tag
        self.tags = [[] for _ in range(len(tag_pdf))]

        self.users = []
        # Generate all the user nodes
        for i in range(n):
            # Determine the tag of the user
            tag = 0
            u = np.random.uniform()
            while u > tag_cdf[tag]:
                tag += 1
            
            self.users.append(self.create_user(self, i, tag))
            self.tags[tag].append(i)
    
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

    def create_user(self, model, i, tag):
        return user(model, i, tag)

    def step(self):

        for user in self.users:
            user.step()

    def run(self, t):

        # t is the number of timesteps
        for _ in range(t):
            self.step()


if __name__ == '__main__':

    n = 10
    test = system(n, 'tags.txt')
    
    for tag, users in enumerate(test.tags):
        ids = []
        for user in users:
            ids.append(user)
        print(tag, ids)
    
    test.run(10)
    print('finished')
