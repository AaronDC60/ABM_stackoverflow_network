import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class user:

    def __init__(self, i):

        # ID
        self.id = i

        # Cost to ask/answer question
        self.c_ask = np.random.randint(1, 5)
        self.c_answer = np.random.randint(1, 5)

        # Willingness to ask/answer question
        self.curiosity = np.random.randint(1, 5)
        self.satisfaction = np.random.randint(1, 5)

        # Interactiveness
        self.interact = np.random.uniform()
    
        # Neighbors in the network
        self.neighbors = []

        # Own questions
        self.my_questions = []
        # Visible questions (from neighbors)
        self.vis_questions = []
    
    def step(self):

        # Evaluate previous questions
        if self.neighbors:
            for q in self.my_questions:
                # Only evaluate the question after everyone had one chance to upvote the question
                if q.age_question == 0:
                    # Change curiosity based on the percentage of upvotes on question
                    self.curiosity *= 2 * (q.upvotes_question / len(self.neighbors))
                    q.age_question += 1

                if q.responder:
                    if q.age_answer == 1:
                        
                        # Upvote answer
                        u = np.random.uniform()
                        if u < self.interact:
                            q.upvotes_answer += 1

                        # Change the satisfaction of the responder based on the percentage of upvotes on the answer
                        q.responder.satisfaction *= 2 * (q.upvotes_answer / len(q.responder.neighbors))
                        # Remove evaluated questions
                        self.my_questions.remove(q)

                    q.age_answer += 1

        # Determine if user will ask a question
        if self.curiosity > self.c_ask:
            q = question(self.id)
            self.my_questions.append(q)
            # Add question to the visible question of the neighbors
            for user in self.neighbors:
                user.vis_questions.append(q)

        # Check if there are visible questions
        for q in self.vis_questions:

            # Upvote question
            u = np.random.uniform()
            if u < self.interact:
                q.upvotes_question += 1

            if q.responder is None:
                # Determine if user will answer the question
                if self.satisfaction > self.c_answer:
                    q.responder = self
                    q.age_answer = 0
                    self.vis_questions.remove(q)
            
            else:
                # Upvote answer
                u = np.random.uniform()
                if u < self.interact:
                    q.upvotes_answer += 1
                self.vis_questions.remove(q)


class question:

    def __init__(self, id):
        
        # For now we have one answer per question

        self.asker = id
        self.responder = None

        self.upvotes_question = 0
        self.upvotes_answer = 0

        self.age_question = 0
        self.age_answer = None

        # Future
    
        # Tag
        # Number of answers
        # (Reputation of asker)

class model:

    def __init__(self, n, p):
        # n is the total number of agents

        # for now we start with a random network
        # p is the probability of a link between 2 nodes in the network

        self.users = []

        # Generate all the nodes
        for i in range(n):
            self.users.append(self.create_user(i))
        
        # Generate the links between the nodes
        for i in range(n):
            for j in range(i + 1, n):
                # Check if there is a link between user i and j
                u = np.random.uniform()
                if u < p:
                    # Create link
                    self.users[i].neighbors.append(self.users[j])
                    self.users[j].neighbors.append(self.users[i])
        
        # Variable to store network
        self.network = self.create_network()

    def create_user(self, i):
        return user(i)

    def step(self):

        for user in self.users:
            user.step()
    
    def create_network(self):

        G = nx.Graph()

        # Create nodes
        for user in self.users:
            G.add_node(user.id, q=user.curiosity/user.c_ask, a=user.satisfaction/user.c_answer, interact=user.interact)
        
        # Create edges
        for i in range(len(self.users)):
            neighbors = self.users[i].neighbors
            for user in neighbors:
                if user.id > self.users[i].id:
                    G.add_edge(self.users[i].id, user.id)

        return G
    
    def plot_network(self, G, attribute, alpha=100):

        size = nx.get_node_attributes(G, attribute).values()
        nx.draw_networkx(G, with_labels=True, node_size=[i * alpha for i in size], node_color = 'red', pos=nx.spring_layout(G, seed=100))
        plt.show()

    def run(self, t):

        # t is the number of timesteps
        for _ in range(t):
            self.step()


if __name__ == '__main__':

    test = model(10, 0.3)

    for user in test.users:
        ids = []
        for node in user.neighbors:
            ids.append(node.id)
        print(user.id, ids)

    test.run(10)
    print('finished')
