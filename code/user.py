"""Module containing the entities of the model (user, question and answer)."""

# Imports
import numpy as np


class user:
    """
    A Stack overflow user.

    Attributes
    ----------
    system : .system
        framework of the model
    id : int
        user id
    tag : int
        tag of user
    reputation : int
        reputation of the user
    p_ask : float
        probability to ask a question
    p_answer : float
        probability to answer a question
    p_interact : float
        probability to upvote a question/answer
    p_active : float
        probability of being active on the site
    vis_questions : list
        all the questions that can be seen by this user
    my_questions : list
        all the questions asked by this user
    upvote_bias : int
        number of upvotes the user is satisfied with
    n_questions_asked : int
        number of questions asked by the user
    n_questions_answered : int
        number of questions answered by the user
    n_questions_upvoted : int
        number of questions upvoted by the user
    n_answers_upvoted : int
        number of answers upvoted by the user
    p_ask_begin : float
        begin probability of asking
    p_answer_begin : float
        begin probability of answering
    p_interact_begin : float
        begin probability of upvoting
    p_active_begin : float
        begin probability of being active

    Methods
    -------
    ask_question()
        Generate a question.
    answer_question(q)
        Generate an answer.
    upvote(interaction, upvote)
        Check if question/answer is upvoted by the user.
    update_p(p, n_upvotes, bias)
        Update probability based on the number of upvotes received.
    eval()
        Evaluate a user's questions and corresponding answers.
    step()
        Timestep of a single user.
    """

    def __init__(self, system, i, tag):
        """
        Initialize a Stack overflow user.

        Parameters
        ----------
        system : .system
            model that represents the Stack oveflow framework
        i : int
            id of the user
        tag : int
            community that the user is part of
        """
        # Model
        self.system = system

        # ID
        self.id = i
        self.tag = tag

        # Starting reputation
        self.reputation = 1

        # Probabilities to ask, answer, upvote, be active
        self.p_ask = 0
        self.p_answer = 0
        self.p_interact = 0
        self.p_active = 0

        # Visible questions/answeres (from people with the same tag)
        self.vis_questions = []
        self.my_questions = []

        # Number of upvotes the user is satisfied with
        self.upvote_bias = system.upvote_bias

        # Data storage
        self.n_questions_asked = 0
        self.n_questions_answered = 0

        self.n_questions_upvoted = 0
        self.n_answers_upvoted = 0

        self.p_ask_begin = 0
        self.p_answer_begin = 0
        self.p_interact_begin = 0
        self.p_active_begin = 0

    def ask_question(self):
        """Generate a question."""
        u = np.random.uniform()
        if u < self.p_ask:
            q = question(self.id, self.tag)
            self.my_questions.append(q)
            self.system.questions.append(q)

            # Make the question visible for all active people with the same tag
            for id in self.system.tags[self.tag]:
                if id != q.asker:
                    u = np.random.uniform()
                    if u < self.system.users[id].p_active:
                        self.system.users[id].vis_questions.append(q)
            self.n_questions_asked += 1

    def answer_question(self, q):
        """
        Generate an answer.

        Parameters
        ----------
        q : .question
            object of the class question

        Returns
        -------
        outcome : int
            1 if question is answered
            0 otherwise
        """
        outcome = 0
        u = np.random.uniform()

        # Lower probability if the question is already answered
        x = np.log(self.p_answer/(1-self.p_answer))
        x -= len(q.answers)
        p_answer = 1 / (1 + np.exp(-x))

        if u < p_answer:
            a = answer(self.id, q.tag)
            q.answers.append(a)
            self.n_questions_answered += 1
            outcome = 1

        return outcome

    def upvote(self, interaction, upvotes):
        """
        Check if question/answer is upvoted by the user.

        Parameters
        ----------
        interaction : .question or .answer
            object of the class question or answer
        upvotes : int
            number of upvotes given already this round

        Returns
        -------
        upvotes : int
            updated number of upvotes
        """
        # Check if the reputation is high enough to upvote
        if self.reputation >= self.system.upvote_treshold:
            u = np.random.uniform()

            # Lower probability if the user has already upvoted question/answers
            x = np.log(self.p_interact/(1-self.p_interact))
            x -= upvotes
            p_upvote = 1 / (1 + np.exp(-x))

            # Upvote question/answer
            if u < p_upvote:
                interaction.upvotes.append(self.id)
                upvotes += 1
                if type(interaction) == question:
                    self.n_questions_upvoted += 1
                    id = interaction.asker
                else:
                    self.n_answers_upvoted += 1
                    id = interaction.responder

                # Increase the reputation
                self.system.users[id].reputation += 10

        return upvotes

    def update_p(self, p, n_upvotes, bias):
        """
        Update probability based on the number of upvotes received.

        Parameters
        ----------
        p : float
            current probability
        n_upvotes : int
            number of upvotes
        bias : int
            min. number of upvotes a user is satisfied with

        Returns
        -------
        new_p : float
            new probability
        """
        # Sensitivity coefficient
        coeff = 0.1

        diff = n_upvotes - bias
        # Limit the difference between -5 and 5
        diff = np.max((diff, -5))
        diff = np.min((diff, 5))

        # Inverse sigmoid function of the probability
        x = np.log(p/(1-p))
        x += (coeff * diff)

        # Calculate new probability using sigmoid function
        new_p = 1/(1+np.exp(-x))

        return new_p

    def eval(self):
        """Evaluate a user's questions and corresponding answers."""
        for q in self.my_questions[::-1]:
            q.age += 1

            # Give every user the chance to upvote all the answers
            if q.age == 2:
                # Update probability of asking and being active
                self.p_ask = self.update_p(self.p_ask, len(q.upvotes), self.upvote_bias)
                self.p_active = self.update_p(self.p_active, len(q.upvotes), self.upvote_bias)

                if q.answers:
                    max = q.answers[0]
                    for a in q.answers:
                        if len(a.upvotes) > len(max.upvotes):
                            max = a

                        # Update probability of answering and being active
                        user = self.system.users[a.responder]
                        user.p_answer = self.update_p(user.p_answer, len(a.upvotes), user.upvote_bias)
                        user.p_active = self.update_p(user.p_active, len(a.upvotes), user.upvote_bias)

                    # Increase the reputation of the user that gave the answer with the most upvotes
                    self.system.users[max.responder].reputation += 15

                else:
                    # If there was no answer on the question, decrease reputation of asker (downvote)
                    self.reputation -= 2
                    self.reputation = np.max((self.reputation, 1))

                self.my_questions.remove(q)

    def step(self):
        """Timestep of a single user."""
        # Evaluate previous questions
        self.eval()

        # Determine if user will ask a question
        self.ask_question()

        # Sort the visible questions based on upvotes
        self.vis_questions.sort(key=lambda x: len(x.upvotes), reverse=True)
        q_upvoted = 0
        for q in self.vis_questions:
            a_upvoted = 0
            # Upvote question
            q_upvoted = self.upvote(q, q_upvoted)
            # Answer question
            answered = self.answer_question(q)
            if not answered:
                q.answers.sort(key=lambda x: len(x.upvotes), reverse=True)
                # Upvote answers
                for a in q.answers:
                    a_upvoted = self.upvote(a, a_upvoted)
        # Remove questions from the visible list
        self.vis_questions = []


class question:
    """
    Represents a question asked by a user.

    Attributes
    ----------
    asker : int
        id of the user that asked the question
    tag : int
        tag (topic) of the question
    age : int
        number of timesteps the question exists
    upvotes : list
        ids of the users that have upvoted the question
    answers : list
        all the answers that were given on this question
    """

    def __init__(self, id, tag):
        """
        Initialize a question.

        Parameters
        ----------
        id : int
            id of the user that asked the question
        tag : int
            tag (topic) of the question
        """
        self.asker = id
        self.tag = tag
        self.age = 0

        self.upvotes = []
        self.answers = []


class answer:
    """
    Represents an answer given by a user.

    Attributes
    ----------
    responder : int
        id of the user that gave the answer
    tag : int
        tag (topic) of the question/answer
    upvotes : list
        ids of the users that have upvoted the answer
    """

    def __init__(self, id, tag):
        """
        Initialize an answer.

        Parameters
        ----------
        id : int
            id of the user that answered the question
        tag : int
            tag (topic) of the question/answer
        """
        self.responder = id
        self.tag = tag

        self.upvotes = []
