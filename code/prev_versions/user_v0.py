"""Class that represents a user on Stack overflow."""

# Imports
import numpy as np
from scipy.stats import norm

class user:

    def __init__(self, system, i, tag):

        # Model
        self.system = system

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

    def ask_question(self):
        """Generate a question."""
        q = question(self.id, self.tag)
        self.my_questions.append(q)

        # Make the question visible for all people with the same tag
        for id in self.system.tags[self.tag]:
            if id != q.asker:
                self.system.users[id].vis_questions.append(q)
        self.n_questions_asked += 1

    def answer_question(self, q):
        """
        Generate an answer.

        Parameters
        ----------
        q : .question
            object of the class question
        """
        u = np.random.uniform()

        # Lower probability if the question is already answered
        x = np.log(self.p_answer/(1-self.p_answer))
        x -= len(q.answers)
        p_answer = 1 / (1 + np.exp(-x))

        # Answer question
        if u < p_answer:
            a = answer(self.id, q.tag)
            q.answers.append(a)

            # Make the answer visible for all people with the same tag
            for id in self.system.tags[q.tag]:
                if id != a.responder:
                    self.system.users[id].vis_answers.append(a)
            self.n_questions_answered += 1

    def upvote(self, interaction):
        """
        Check if question/answer is upvoted.

        Parameters
        ----------
        interaction : .question or .answer
            object of the class question or answer
        """
        # Minimum reputation for upvoting is 15
        if self.reputation >= 15:
            u = np.random.uniform()

            # Higher probability if the question/answer has more upvotes
            x = np.log(self.p_interact/(1-self.p_interact))
            x += interaction.n_upvotes
            p_upvote = 1 / (1 + np.exp(-x))

            # Upvote question/answer
            if u < p_upvote:
                interaction.n_upvotes += 1
                if type(interaction) == question:
                    self.n_questions_upvoted += 1
                    id = interaction.asker
                else:
                    self.n_answers_upvoted += 1
                    id = interaction.responder
            
                # Increase the reputation
                self.system.users[id].reputation += 10

    def eval(self):
        """Evaluate user's questions and corresponding answers."""
        for q in self.my_questions[::-1]:
            q.age += 1

            # Give every user the chance to upvote all the answers
            if q.age == 2:                
                if q.answers:
                    # Increase the reputation of the user that gave the answer with the most upvotes
                    max = q.answers[0]
                    for a in q.answers:
                        if a.n_upvotes > max.n_upvotes:
                            max = a

                    self.system.users[max.responder].reputation += 15

    def step(self):
        """Timestep of a single user."""
        # Evaluate previous questions
        self.eval()

        # Determine if user will ask a question
        u = np.random.uniform()
        if u < self.p_ask:
            self.ask_question()

        # Check if there are visible questions
        for q in self.vis_questions:
            # Upvote question?
            self.upvote(q)
            # Answer question?
            self.answer_question(q)
        # Remove questions from the visible list
        self.vis_questions = []

        for a in self.vis_answers:
            # Upvote answer?
            self.upvote(a)
        # Remove answers from the visible list    
        self.vis_answers = []

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
        