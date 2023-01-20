"""Class that represents a user on Stack overflow."""

# Imports
import numpy as np
from scipy.stats import norm

class user:

    def __init__(self, model, i, tag):

        # Model
        self.system = model

        # ID
        self.id = i
        self.tag = tag

        # Starting reputation is 1
        self.reputation = 1

        # Probabilities to ask, answer, upvote
        # Drawn from a normal distribution
        self.p_ask = norm.cdf(np.random.normal())
        self.p_answer = norm.cdf(np.random.normal())
        self.p_interact = norm.cdf(np.random.normal())

        # Visible questions/answeres (from people with the same tag)
        self.vis_answers = []
        self.vis_questions = []
        self.my_questions = []

        # Data storage

        self.n_questions_asked = 0
        self.n_questions_answered = 0

        self.n_questions_upvoted = 0
        self.n_answers_upvoted = 0

class question:

    def __init__(self, id, tag):

        self.asker = id
        self.tag = tag
        self.age = 0

        self.n_upvotes = 0
        self.answers = []

class answer:

    def __init__(self, id, tag):

        self.responder = id
        self.tag = tag

        self.n_upvotes = 0