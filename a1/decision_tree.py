#import numpy to load data
from numpy import genfromtxt

#load data from file location
test_data = genfromtxt('data_1.csv', delimiter = ',', encoding = "utf8", dtype = None)

#name of columns
features = ['pclass', 'age', 'gender', 'survived']

#remove the header row from dataset
test_data = test_data[2:]

#function to return the number of unique values in a column, passed by column number (refer to column_names for index)
def unique_values(test_data, column_index):
    return set([test[column_index] for test in test_data])

#Returns the number of rows which fall under each label (last column, 'survived' in this case)
#For given data set, returns a dictionary: count = {'yes': 200, 'no': 300}  (numbers are arbitrary) 
def calculate_freq(rows):
    count = {}
    for row in rows:
        label = row[-1]
        if label not in count:
            count[label] = 0
        count[label] += 1
    return count

#Returns true if the passed parameter is a numeric value or not
def is_numeric(x):
    return isinstance(x, (int, float))


#Class to store a question based on the template (feature_index, val) which checks if the column indexed by feature_index has:
# 1. value >= val if val is a numeric datatype
# 2. value == val, otherwise
class Question:
    
    def __init__(self, feature_index, value):
        self.feature_index = feature_index
        self.value = value
    
    def match(self, example):
        x = example[self.feature_index]
        if is_numeric(x):
            return x >= self.value
        else:
            return x == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
        features[self.feature_index], condition, str(self.value))

#Partitions the data set into 2 distinct sets on the basis of a passed question
def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

#Calculates the gini index of a set of rows identified by the last value (labels) of each row
def GINI(rows):
    counts = calculate_freq(rows)
    impurity = 1
    for label in counts:
        prob = counts[label] / float(len(rows))
        impurity -= prob**2
    return impurity


#Calculates the information gain if a dataset is divided into 2 sets, left and right
def info_gain(left, right, current_gini):
    p = float(len(left)) / (len(left) + len(right))
    return current_gini - p * GINI(left) - (1-p)*GINI(left)

#Determines the most optimal question to ask which leads to the maximum information gain
def find_best_split(rows):
    best_gain = 0
    best_question = None
    current_gini = GINI(rows)
    n_features = len(rows[0]) - 1
    
    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) ==0 or len(false_rows) == 0:
                continue
            
            gain = info_gain(true_rows, false_rows, current_gini)
            
            if gain >= best_gain:
                best_gain, best_question = gain, question
            
    return best_gain, best_question

#Class to store the leaf nodes of a decision tree
class Leaf:
    def __init__(self,rows):
        self.predictions = calculate_freq(rows)

#Class to store a non-leaf node of the decision tree
class Decision_Node:
    def __init__(self,
                question,
                true_branch,
                false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

#function to build tree with input parameter as the training data
def build_tree(rows):
    gain, question = find_best_split(rows)
    
    if gain == 0:
        return Leaf(rows)
    
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    
    return Decision_Node(question, true_branch, false_branch)

#Function to print leaf nodes
def print_leaf(count):
    total = sum(count.values()) * 1.0
    prob = {}
    for label in count.keys():
        prob[label] = str(int(count[label] / total * 100)) + "%"
    return prob

#Function to print tree recursively
def print_tree(node, spacing=""):
   
    if isinstance(node, Leaf):
        print (spacing + " Predict", print_leaf(node.predictions))
        return

    print (spacing + "•" + str(node.question))

    print (spacing + ' ⮕ True:')
    print_tree(node.true_branch, spacing + "   ")

    print (spacing + ' ⮕ False:')
    print_tree(node.false_branch, spacing + "   ")


print("\n•Machine Learning, CS60050: Assignment 1: DECISION TREES")
print("•Pankaj Mishra, 17EC30043\n")

d_tree = build_tree(test_data)
print_tree(d_tree)

print("\n---END---\n")
