import numpy as np


class Node:

    def __init__(self, threshold, feature_idx, depth):
        self.left_child = None
        self.right_child = None
        self.threshold = threshold
        self.feature_idx = feature_idx
        self.depth = depth
        self.class_dist = None
        self.prob_dist = None
        self.score = None
        self.label = None
        self.cost = 1.0


class DecisionTree:

    def __init__(self, train_data, test_data, cost_data=None, impurity_measure="entropy", prune_length=None,
                 use_cost=False, cost_measure="nunez", cost_weight=0.01, prune_threshold=None):
        self.train_data = train_data
        self.test_data = test_data
        self.cost_metrics = cost_data
        self.features = list(self.cost_metrics.keys())
        self.features.append("x21")
        self.prune_length = prune_length
        self.prune_threshold = prune_threshold
        self.cost_enabled = use_cost
        if impurity_measure == "gini":
            self.measure = "gini"
        elif impurity_measure == "entropy":
            self.measure = "entropy"

        if cost_measure == "nunez":
            self.cost_measure = "nunez"
            self.cost_weight = cost_weight
        elif cost_measure == "tan":
            self.cost_measure = "tan"

        self.used_features = []

        self.root = self.grow_tree()

    @staticmethod
    def get_node_occurrences(labels):
        occurrences = np.zeros(3, dtype=int)  # 3 is the number of classes for this classifier
        classes, counts = np.unique(labels, return_counts=True)
        label_count = np.sum(counts)
        for class_idx in range(classes.size):
            occurrences[int(classes[class_idx]) - 1] = counts[class_idx]
        # Handling division by 0
        prob_dist = occurrences / label_count if label_count != 0 else occurrences
        return occurrences, prob_dist

    def get_node_entropy(self, labels):
        _, prob_dist = self.get_node_occurrences(labels)
        log_dist = np.log2(prob_dist, where=[prob != 0 for prob in prob_dist])
        entropy = -np.sum(np.multiply(prob_dist, log_dist))
        return entropy

    def get_node_gini(self, labels):
        _, prob_dist = self.get_node_occurrences(labels)
        gini = 1 - np.sum(np.square(prob_dist))
        return gini

    def get_node_impurity(self, labels):
        if self.measure == "entropy":
            return self.get_node_entropy(labels)
        elif self.measure == "gini":
            return self.get_node_gini(labels)
        else:
            return self.get_node_misclassification_impurity(labels)

    def get_node_misclassification_impurity(self, labels):
        _, prob_dist = self.get_node_occurrences(labels)
        misclassification_impurity = 1 - np.max(prob_dist)
        return misclassification_impurity

    def get_split_impurity(self, data, threshold, feature_idx):
        feature_data = data[:, feature_idx]
        data_split = feature_data <= threshold
        labels = data[:, -1]
        left_subtree = labels[data_split]
        right_subtree = labels[~data_split]
        left_prob = len(left_subtree) / len(labels)
        right_prob = 1 - left_prob
        # Handling 0log0 case
        right_impurity = self.get_node_entropy(right_subtree) if right_prob != 0 else 0
        left_impurity = self.get_node_impurity(left_subtree) if left_prob != 0 else 0
        return right_prob * right_impurity + left_prob * left_impurity

    def get_feature_cost(self, feature_idx):
        if feature_idx in self.used_features:
            return 10 ** (-8)
        else:
            if feature_idx == 20:
                cost = 1.0
                if 19 not in self.used_features:
                    cost += self.cost_metrics[self.features[19]]
                    self.used_features.append(19)
                if 18 not in self.used_features:
                    cost += self.cost_metrics[self.features[18]]
                    self.used_features.append(18)
                return cost
            else:
                return self.cost_metrics[self.features[feature_idx]]

    def get_cost_split(self, data, prev_impurity):
        no_of_features = data.shape[1] - 1
        best_score = float("-inf")
        split_idx = -1
        split_threshold = -1
        best_cost = float("inf")
        for feature_idx in range(no_of_features):
            possible_splits = np.unique(data[:, feature_idx])
            if feature_idx in (0, 16, 17, 18, 19, 20):
                possible_splits = (possible_splits[1:] + possible_splits[:-1]) / 2
            for threshold in possible_splits:
                impurity = self.get_split_impurity(data, threshold, feature_idx)
                gain = prev_impurity - impurity
                cost = self.get_feature_cost(feature_idx)
                score = 0
                if self.cost_measure == "tan":
                    score = (gain ** 2) / cost
                else:
                    score = ((2 ** gain) - 1) / ((cost + 1) ** self.cost_weight)
                if score > best_score:
                    best_score = score
                    split_idx = feature_idx
                    self.used_features.append(feature_idx)
                    split_threshold = threshold
                    best_cost = cost
        return best_score, split_idx, split_threshold, best_cost

    def get_best_split(self, data):
        no_of_features = data.shape[1] - 1
        best_impurity = 1.01
        split_idx = -1
        split_threshold = -1
        for feature_idx in range(no_of_features):
            possible_splits = np.unique(data[:, feature_idx])
            if feature_idx in (0, 16, 17, 18, 19, 20):
                possible_splits = (possible_splits[1:] + possible_splits[:-1]) / 2
            for threshold in possible_splits:
                impurity = self.get_split_impurity(data, threshold, feature_idx)
                if impurity < best_impurity:
                    best_impurity = impurity
                    split_idx = feature_idx
                    split_threshold = threshold
        return best_impurity, split_idx, split_threshold

    def split_tree(self, node, data):
        node_impurity = self.get_node_impurity(data[:, -1])
        if node_impurity == 0 or node.depth == self.prune_length:
            class_dist, prob_dist = self.get_node_occurrences(data[:, -1])
            node.label = np.argmax(prob_dist) + 1
            node.score = node_impurity
            node.class_dist = class_dist
            node.prob_dist = prob_dist
            return
        cost = 1.0
        if self.cost_enabled:
            split_score, split_idx, split_threshold, cost = self.get_cost_split(data, node_impurity)
        else:
            split_score, split_idx, split_threshold = self.get_best_split(data)

        if self.prune_threshold:
            if not self.cost_enabled and node_impurity - split_score < self.prune_threshold:
                class_dist, prob_dist = self.get_node_occurrences(data[:, -1])
                node.label = np.argmax(prob_dist) + 1
                node.score = node_impurity
                node.class_dist = class_dist
                node.prob_dist = prob_dist
                return

        split = data[:, split_idx] <= split_threshold
        # Update node variables
        node.score = node_impurity
        node.threshold = split_threshold
        node.feature_idx = split_idx
        node.cost = cost
        left_data = data[split]
        left_occurrences, left_prob_dist = self.get_node_occurrences(left_data[:, -1])
        left_node = Node(None, None, node.depth + 1)
        left_node.class_dist = left_occurrences
        left_node.prob_dist = left_prob_dist
        node.left_child = left_node
        # Create right child
        right_data = data[~split]
        right_occurrences, right_prob_dist = self.get_node_occurrences(right_data[:, -1])
        right_node = Node(None, None, node.depth + 1)
        right_node.class_dist = right_occurrences
        right_node.prob_dist = right_prob_dist
        node.right_child = right_node
        # Pre-order traversal for tree construction
        self.split_tree(left_node, left_data)
        self.split_tree(right_node, right_data)

    def grow_tree(self):
        root = Node(None, None, 0)
        occurrences, _ = self.get_node_occurrences(self.train_data[:, -1])
        root.class_dist = occurrences
        self.split_tree(root, self.train_data)
        return root

    def test_sample(self, sample):
        node = self.root
        while not node.label:
            feature_idx = node.feature_idx
            threshold = node.threshold
            if sample[feature_idx] <= threshold:
                node = node.left_child
            else:
                node = node.right_child
        return node.label

    def evaluate_model(self):
        confusion_matrix = np.zeros((3, 3), dtype=int)
        for sample_idx in range(self.test_data.shape[0]):
            sample = self.test_data[sample_idx, :]
            label = self.test_sample(sample)
            confusion_matrix[int(sample[-1]) - 1, label - 1] += 1
        return confusion_matrix


def split_data(data, feature_idx, threshold):
    feature_values = data[:, feature_idx]
    split = feature_values < threshold
    # Return left node and right node
    return data[split], data[~split]

