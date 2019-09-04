#17EC30043 Pankaj Mishra Assignment 2 

import pandas

train_d = pandas.read_csv('data2_19.csv', sep=',')
test_d = pandas.read_csv('test2_19.csv', sep=',')

#formatting data

train_data = []
for index,row in train_d.iterrows():
    train_data.append(row[0].split(','))

test_data = []
for index, row in test_d.iterrows():
    test_data.append(row[0].split(','))

final_train_data = []
for data in train_data:
    final_train_data.append([int(i) for i in data])

final_test_data = []
for data in test_data:
    final_test_data.append([int(i) for i in data])


def test(train_data, test_data):
    prob_true = 0
    positive_examples = 0
    negative_examples = 0
    total_examples = len(train_data)
    
    for ex in train_data:
        if ex[0] == 1:
            positive_examples += 1
        else:
            negative_examples += 1
    
    print("Parsing training data: \n")
    print(">> number of positive examples = " , positive_examples)
    print(">> number of negative examples = " , negative_examples)
    print(">> number of total examples = " , total_examples)
    
    
    prob_true = positive_examples * 1.0 / total_examples
    prob_false = 1.0 - prob_true
    
    #print("Probability of positive test case = ", prob_true)
    #print("Probability of negaive test case = " , prob_false)
    
    correct_predictions = 0
    total_predictions = len(test_data)
    
    print("\nStarting classification...\n")
    for test in test_data:
        
        prob_yes = 1.0
        prob_no = 1.0
        
        #print("currently testing hypothesis: ")
        #print(test)
                
        for attribute_index in range(1, 7):
            
            #print("currently testing attribute number: ", attribute_index)
            
            yes_cases = 0
            no_cases = 0
            
            match_yes_cases = 0
            match_no_cases = 0
            
            for case in train_data:
                #print("training data: ")
                #print(case)
                
                if case[0] == 1:
                    yes_cases += 1
                    #print("positive example, number of positive examples = ", yes_cases)
                    if (test[attribute_index] == case[attribute_index]):
                        match_yes_cases += 1
                        #print("attribute , matched positive examples = ", match_yes_cases)
                
                if case[0] == 0:
                    no_cases += 1
                    #print("negative example, number of negative examples = ", no_cases)
                    if (test[attribute_index] == case[attribute_index]):
                        match_no_cases += 1
                        #print("attribute matched, matched negative examples = ", match_no_cases)
                        
            #print("total yes cases = " , yes_cases, " matched yes cases = " , match_yes_cases)
            #print("total no cases = " , no_cases, " matched no cases = " , match_no_cases)
            
            
            match_yes_cases += 1
            match_no_cases += 1
            
            yes_cases += 5
            no_cases += 5
            
            py = (match_yes_cases * 1.0) / yes_cases
            pn = (match_no_cases * 1.0) / no_cases
            
            #print("p of yes = ", py, " p of no = " , pn)
                
            prob_yes *= py
            prob_no *= pn
            
            #print("P(yes) = ", prob_yes, " P(no) = ", prob_no)
        
        prob_yes *= prob_true
        prob_no *= prob_false
        
        if (prob_yes >= prob_no):
            #predict yes
            if (test[0] == 1):
                print("Correct prediction!")
                correct_predictions += 1
            else:
                print("Wrong prediction!")
        else:
            if (test[0] == 0):
                print("Correct prediction!")
                correct_predictions += 1
            else:
                print("Wrong prediction!")
                
    
    print("\nClassification finished...\n")
        
    accuracy = correct_predictions / total_predictions
    print("Accuracy of classifier = " , accuracy * 100, '%')

test(final_train_data, final_test_data)