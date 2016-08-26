from random import uniform, choice


class QLearningAgent:
    def __init__(self, extractor, actions, learning=True):
        self.weights = {}
        self.extractor = extractor
        self.qvalues = {}
        self.learning = learning
        self.actions = actions

        # Learning rate
        self.alpha = 0.2
        # Reward Discount
        self.discount = 0.8
        # Probability of random action
        self.epsilon = 0.2

    def getQValue(self, state, action):
        features = self.extractor.getFeatures(state, action)
        q_value = 0.0
        for key in features:
            feature_value = features[key]
            if key in self.weights:
                q_value += self.weights[key] * feature_value
            else:
                self.weights[key] = 0.0
                q_value += self.weights[key] * feature_value
        return q_value

    def computeValueFromQValues(self, state):
        value_list = []
        for action in self.getLegalActions():
            value_list.append(self.getQValue(state, action))
        if len(value_list) == 0:
            value = 0.0
        else:
            value = max(value_list)
        return value

    def computeActionFromQValues(self, state):
        max_value = -float('inf')
        best_action = None
        for action in self.getLegalActions():
            actual_value = self.getQValue(state, action)
            if actual_value > max_value:
                max_value = actual_value
                best_action = action

        return best_action

    def getAction(self, state):
        random = uniform(0, 1)
        if random <= self.epsilon:
            action = choice(self.getLegalActions())
        else:
            action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, reward, nextState=None):

        if nextState is None:
            nextStateValue = 0
        else:
            nextStateValue = self.computeValueFromQValues(nextState)

        qvalue = self.getQValue(state, action)
        diff = (reward + self.discount * nextStateValue) - (qvalue)

        features = self.extractor.getFeatures(state, action=action)
        for key in self.weights:
            new_weight = self.weights[key] + self.alpha * diff * features[key]
            self.weights[key] = new_weight

    def getLegalActions(self):
        return self.actions