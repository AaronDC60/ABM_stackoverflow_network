"""Test file for the model that represents the Stack overflow network."""

# Imports
import numpy as np
import unittest
import model

class test_model(unittest.TestCase):

    def setUp(self):
        # Initialize several models with different settings
        self.system1 = model.system(20, 'tags.txt')
        self.system2 = model.system(20, 'tags.txt', treshold=5, bias=4, distr=[[0.4, 0.75], [None, None], [2, None], [None, 0.3]])

        # Create one user in every model
        np.random.seed(0)
        self.user1 = self.system1.create_user(0)
        self.user2 = self.system2.create_user(1)

        # Create several users with the same tag
        self.user3 = self.system2.create_user(0)
        self.user4 = self.system2.create_user(1)
        self.user5 = self.system2.create_user(2)
        self.user6 = self.system2.create_user(3)
        self.user7 = self.system2.create_user(4)
        self.user3.tag = 5
        self.user4.tag = 5
        self.user5.tag = 5
        self.user6.tag = 4
        self.user7.tag = 5

        # Add users to the system
        self.system2.users = [self.user3, self.user4, self.user5, self.user6, self.user7]
        self.system2.tags = [[],[],[], [], [self.user6.id], [self.user3.id, self.user4.id, self.user5.id, self.user7.id]]

        # Set some probabilities to have certain interactions
        self.user3.p_ask = 0.999
        self.user3.p_ask_begin = 0.999
        self.user4.p_active = 0.999
        self.user4.p_active_begin = 0.999
        self.user5.p_active = 0
        self.user6.p_active = 1
        self.user7.p_active = 0.999
        self.user7.p_active_begin = 0.999

        # Simulate interactions
        self.user3.ask_question()

    def test_user_settings(self):
        # Test the tag of the user
        self.assertEqual(self.user1.tag, 4)
        self.assertEqual(self.user2.tag, 5)

        # Test the id of the user
        self.assertEqual(self.user1.id, 0)
        self.assertEqual(self.user2.id, 1)

        # Test the upvote bias
        self.assertEqual(self.user1.upvote_bias, 12)
        self.assertEqual(self.user2.upvote_bias, 4)

        # Check if probabilities were drawn from the right distribution
        self.assertEqual(self.user2.p_ask, 0.42230568454271344)
        self.assertEqual(self.user2.p_answer, 0.8917730007820798)
        self.assertEqual(self.user2.p_interact, 0.8956153681081356)
        self.assertEqual(self.user2.p_active, 0.3834415188257777)

        # Check if starting reputation is 1
        self.assertEqual(self.user1.reputation, 1)
    
    def test_step(self):
        # Test if users are added during a timestep
        self.system1.step()
        self.assertEqual(len(self.system1.users), 20)

    def test_asking(self):
        # Test if the dynamics of asking a question are correct

        # Question is asked
        self.assertEqual(self.user3.n_questions_asked, 1)
        self.assertEqual(len(self.user3.my_questions), 1)
        self.assertEqual(len(self.system2.questions), 1)

        # User 4 sees question
        self.assertEqual(len(self.user4.vis_questions), 1)
        # User 5 does not see question (non-active)
        self.assertEqual(len(self.user5.vis_questions), 0)
        # User 6 does not see question (different tag)
        self.assertEqual(len(self.user6.vis_questions), 0)
    
    def test_answering(self):
        # Test if the dynamics of answering a question are correct
        self.user4.p_answer = 0.01
        outcome = self.user4.answer_question(self.user4.vis_questions[0])

        # Question is not answered
        self.assertEqual(outcome, 0)
        self.assertEqual(self.user4.n_questions_answered, 0)
        self.assertEqual(len(self.user3.my_questions[0].answers), 0)

        self.user4.p_answer = 0.999
        outcome = self.user4.answer_question(self.user4.vis_questions[0])

        # Question is answered        
        self.assertEqual(outcome, 1)
        self.assertEqual(self.user4.n_questions_answered, 1)
        self.assertEqual(len(self.user3.my_questions[0].answers), 1)

    def test_upvoting(self):
        # Test if the dynamics of upvoting a question/answer are correct
        self.user4.p_interact = 1
        outcome = self.user4.upvote(self.user4.vis_questions[0], 0)

        # Question is not upvoted (below upvote treshold)
        self.assertEqual(outcome, 0)
        self.assertEqual(len(self.user3.my_questions[0].upvotes), 0)
        self.assertEqual(self.user4.n_questions_upvoted, 0)

        self.user4.reputation = 100
        self.user4.p_interact = 0.01
        outcome = self.user4.upvote(self.user4.vis_questions[0], 0)

        # Question is not upvoted (p is too low)
        self.assertEqual(outcome, 0)
        self.assertEqual(len(self.user3.my_questions[0].upvotes), 0)
        self.assertEqual(self.user4.n_questions_upvoted, 0)

        self.user4.p_interact = 0.999
        outcome = self.user4.upvote(self.user4.vis_questions[0], 0)

        # Question is upvoted
        self.assertEqual(outcome, 1)
        self.assertEqual(self.user3.my_questions[0].upvotes, [self.user4.id])
        self.assertEqual(self.user4.n_questions_upvoted, 1)
        self.assertEqual(self.user3.reputation, 11)

    def test_feedback(self):
        # Test if the feedback mechanism to update the probabilities is correct
        self.user4.p_answer = 0.999
        self.user4.p_answer_begin = 0.999
        self.user7.p_answer = 0.999
        self.user7.p_answer_begin = 0.999

        self.user3.p_interact = 0.999
        self.user4.p_interact = 0.999
        self.user7.p_interact = 0.999

        # Set reputation high enough to upvote
        self.user3.reputation = 15
        self.user4.reputation = 15
        self.user7.reputation = 15

        # User 4 and 7 answer the question of user 3
        self.user4.answer_question(self.user4.vis_questions[0])
        self.user7.answer_question(self.user7.vis_questions[0])

        # Check if there are 2 answers
        self.assertEqual(len(self.user3.my_questions[0].answers), 2)

        # User 3 and 4 upvote the answer of user 7
        self.user3.upvote(self.user3.my_questions[0].answers[1], 0)
        self.user4.upvote(self.user4.vis_questions[0].answers[1], 0)
        # User 3 upvotes the answer of user 4
        self.user3.upvote(self.user3.my_questions[0].answers[0], 0)

        # Check if upvotes are registered and reputation is adjusted
        self.assertEqual(self.user3.n_answers_upvoted, 2)
        self.assertEqual(self.user4.n_answers_upvoted, 1)
        self.assertEqual(self.user7.n_answers_upvoted, 0)
        self.assertEqual(self.system2.questions[0].answers[0].upvotes, [self.user3.id])
        self.assertEqual(self.system2.questions[0].answers[1].upvotes, [self.user3.id, self.user4.id])
        self.assertEqual(self.user4.reputation, 25)
        self.assertEqual(self.user7.reputation, 35)

        # Change the upvote bias of user 4 and 7
        self.user4.upvote_bias = 1
        self.user7.upvote_bias = 1

        # Evaluate current questions
        self.user3.eval()

        # Probabilities should not change after first evaluation
        self.assertEqual(self.user3.my_questions[0].age, 1)
        self.assertEqual(self.user3.p_ask, self.user3.p_ask_begin)
        self.assertEqual(self.user3.p_active, self.user3.p_active_begin)
        self.assertEqual(self.user4.p_answer, self.user4.p_answer_begin)
        self.assertEqual(self.user4.p_active, self.user4.p_active_begin)
        self.assertEqual(self.user7.p_answer, self.user7.p_answer_begin)
        self.assertEqual(self.user7.p_active, self.user7.p_active_begin)
        # Reputation should not change
        self.assertEqual(self.user3.reputation, 15)
        self.assertEqual(self.user4.reputation, 25)
        self.assertEqual(self.user7.reputation, 35)

        self.user3.eval()

        # Probabilities should have been updated after the second evaluation
        self.assertEqual(self.system2.questions[0].age, 2)
        self.assertEqual(len(self.user3.my_questions), 0)
        # Upvotes < upvote_bias -> decrease probability
        self.assertTrue(self.user3.p_ask < self.user3.p_ask_begin)
        self.assertTrue(self.user3.p_active < self.user3.p_active_begin)
        # Upvotes == upvote_bias -> unchanged probability
        self.assertEqual(np.round(self.user4.p_answer, 3), np.round(self.user4.p_answer_begin, 3))
        self.assertEqual(np.round(self.user4.p_active, 3), np.round(self.user4.p_active_begin, 3))
        # Upvotes > upvote_bias -> increase probability
        self.assertTrue(self.user7.p_answer > self.user7.p_answer_begin)
        self.assertTrue(self.user7.p_active > self.user7.p_active_begin)
        # Most upvotes -> additional reputation points
        self.assertEqual(self.user7.reputation, 50)

if __name__ == '__main__':
    unittest.main()
    