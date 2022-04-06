"""
 Student : Ibrahim Hashmi
      ID : 6352926
Language : Python
     IDE : VSCode
"""

import numpy as np
import math
import random
import statistics

MAX_EPOCH = 10000    #maximum number of epochs

# number of neurons in each layer
INPUT_NEURONS = 4
HIDDEN_NEURONS = 8
OUTPUT_NEURONS = 1

LEARNING_RATE = 0.8 

inputNums=[]
expected_outputs=[]
testing_data = ['0000','0111','1011','0011']
testing_outputs = []



# Obtain training dataset from CSV file
with open('trainingdata.csv',newline='') as csvfile:    
    header = next(csvfile)
    for line in csvfile:
        inputNums.append(line.split(",")[0])
        expected_outputs.append(line.split(",")[2])

training_data = inputNums[:]

# Remove every value in the training dataset that is also in the testing dataset
for j in range(len(testing_data)):
    testing_outputs.append(expected_outputs[training_data.index(testing_data[j])])  # obtain expected outputs for testing dataset
    del expected_outputs[training_data.index(testing_data[j])]
    training_data.remove(testing_data[j])



# Sigmoid function. This is used 
def sigmoid(num):
    x = 1.0 / (1.0 + math.exp(-num))
    return x


# Randomnly initialize the weights 
def initialize_weights():
    W_2 = np.random.uniform(-1,1,(HIDDEN_NEURONS,INPUT_NEURONS))
    W_3 = np.vstack(np.random.uniform(-1,1,(OUTPUT_NEURONS,HIDDEN_NEURONS)))
    return W_2, W_3




# Calculate the action values of an A matrix (either A1, A2 or A3) by applying the activation function
def calculate_A(input,num_nodes):
    a = []

    # for every value in the given matrix, apply sigmoid function to is and append that to a new list 
    for i in range(num_nodes):
        a.append(sigmoid(float(input[i]))) 
    return a



# Cost function: calculate the error of the model using the expected and actual outputs
def cost(actual,expected):
    y = float(expected)
    cost = (float(actual[0])-y)**2  # C = (A^3 - y)^2
    return cost



# Calculate sigmoid prime of every value in a matrix, and return a new matrix filled with those values
def sigmoidPrime(matrix):
    new_matrix = []

    # for every value in the given matrix, calculate sigmoid prime of that value
    for i in range(len(matrix)):
        new_matrix.append(matrix[i]*(1-matrix[i]))

    return np.vstack(new_matrix)    # return as a vertical matrix



'''
PARAMETERS: list[[]] (A_3 matrix), list[] (expected_outputs), list[[]] (A_2 matrix)
RETURN: FLOAT - delta3

DESCRIPTION: This function calculates and returns the gradient for the weight matrix between the hidden layer A2 and the output layer 
A3 using the delta3 formula given in class
'''
def gradient3(A_3,expected_output,A_2): #A_3 = A3 matrix, y = expected output, A_2 = A2 matrix

    y = float(expected_output) 

    # Represent the given matrices as vertical matrices
    A2 = np.vstack(A_2)
    A3 = np.vstack(A_3)

    A2_T = A2.T # transpose of A2

    sigmoidPrime_X3 = sigmoidPrime(A3) 

    a = np.multiply((2*(A3-y)),sigmoidPrime_X3) # 2(A^3 - y) * sigPrime(X^3)
    delta3 = np.matmul(a,A2_T)                  # 2(A^3 - y) * sigPrime(X^3) * (A^1)^T

    return delta3
    

'''
PARAMETERS: list[[]] A_3 matrix, list [[]] (W_3 - weight matrix 3), list[[]] A_2, list[[]] A_1
RETURN: FLOAT - delta2

DESCRIPTION: This function calculates and returns the gradient for the weight matrix between the input layer A1 and the hidden layer 
A2 using the delta2 formula given in class
'''
def gradient2(A_3,expected_output,W_3,A_2,A_1):

    y = float(expected_output) 

    # Represent the given matrices as vertical matrices
    A1 = np.vstack(A_1)
    A2 = np.vstack(A_2)
    A3 = np.vstack(A_3)
    W3 = np.vstack(W_3)

    A1_T = A1.T # transpose of A1
    sigPrime_X3 = sigmoidPrime(A3)  # sigmoid prime of A3
    sigPrime_X2 = sigmoidPrime(A2).T # sigmoid prime of A2 (transposed)

    a = np.multiply(2*(A3-y),sigPrime_X3)   # 2*(A^3 - y) * sigPrime(X^3)
    b = np.multiply(a,W3)                   # 2*(A^3 - y) * sigPrime(X^3) * W^3
    c = np.multiply(b,sigPrime_X2).T        # 2*(A^3 - y) * sigPrime(X^3) * W^3 * sigPrime(A^2)^T)^T
    delta2 = np.matmul(c,A1_T)              #(2*(A^3 - y) * sigPrime(X^3) * W^3 * sigPrime(A^2)^T)^T * (A^1)^T 
    
    return delta2


'''
PARAMETERS: list[] (input_data), list [[]] (W_2 - weight matrix 2), list [[]] (W_3 - weight matrix 3)
RETURN: TUPLE (A3, A2, A1)

DESCRIPTION: This function calculates the activation layers and feeds forward the data from the input action-layer (A1) to the hidden layer (A2), and then from there to 
the output layer (A3) using an activation function. 
'''
def forward_pass(input_data,W_2,W_3):
    A1 = calculate_A(input_data,INPUT_NEURONS) 
    X2 = np.matmul(W_2,A1)
    A2 = calculate_A(X2,HIDDEN_NEURONS) # the hidden layer
    X3 = np.matmul(W_3,A2)
    A3 = calculate_A(X3,OUTPUT_NEURONS) # the output layer
    return A3, A2, A1
    


'''
PARAMETERS: int (max_epochs), list[] (training_inputs), list[] (expected_outputs), float (learning_rate)
RETURN: TUPLE (W_2, W_3) - weight matrices

DESCRIPTION: This function trains the neural network by initializing the weight matrices with random values between -1.0 and 1.0, and inputting 12 training datasets 
through the network per epoch. Each time a training dataset is input, the network determines an output and then checks how correct that output is by calculating the 
downward gradients for each weight matrix, and adjusting the weights within them accordingly. This happens for every epoch until the maximum number of epochs is reached.

Every 200 epochs, the Mean Score Error is calculated and printed along with it's corresponding Epoch No. 
'''
def training(max_epochs,training_inputs,expected_outputs,learning_rate):

    num_of_epochs = 0
    epoch_inc_200 = 0   # 200 epochs increment

    # initialize and assign weight matrices 
    weight_matrices = initialize_weights()
    W_2 = weight_matrices[0] # input-to-hidden layer weight matrix
    W_3 = weight_matrices[1] # hidden-to-output layer weight matrix

    mse = 0

    print("Total Epochs: "+str(max_epochs)+", Learning Rate: "+str(learning_rate))

    while num_of_epochs<=max_epochs:

        error_scores = []
        for i in range(len(training_inputs)):   # for every input in training data

            
            matrices = forward_pass(training_inputs[i],W_2,W_3) # Get A3, A2 and A1
            actual_output = matrices[0] # A3

            # calculate error for current output, and add to list of error scores for calculating mean
            squared_error = cost(actual_output,expected_outputs[i])
            error_scores.append(squared_error)  

            # calculate gradients for both weight matrices
            delta3 = gradient3(actual_output,expected_outputs[i],matrices[1])   # gradient for hidden-to-output layer weight matrix
            delta2 = gradient2(matrices[0],expected_outputs[i], W_3, matrices[1],matrices[2])   # gradient for input-to-hidden layer weight matrix
            

            W_3 = np.subtract(W_3,learning_rate*delta3)  # update weight matrix for output layer

            W_2 = np.subtract(W_2,learning_rate*delta2)  # update weight matrix for hidden layer

            # Uncomment line below to see effectiveness of network's training per each input
            # print("Epoch No. "+str(num_of_epochs)+", Training Dataset: "+str(training_inputs[i])+", Expected Output: "+str(expected_outputs[i])+", Actual Output: "+str(actual_output))  


        mse = statistics.mean(error_scores)

        if epoch_inc_200 == 200:  
            epoch_inc_200 = 0
            print("Epoch No. "+str(num_of_epochs)+", MSE: "+ str(mse))

        epoch_inc_200 += 1
        num_of_epochs+=1

    print("\nNo. of Hidden Nodes: ",HIDDEN_NEURONS,", Learning Rate: ",LEARNING_RATE,", Final MSE: ",mse)
    print("----------------------------")

    for i in range(len(training_inputs)):
        output = forward_pass(training_inputs[i],W_2,W_3)[0]
        print("Training Data Example: "+str(training_inputs[i])+", Expected Output: "+str(expected_outputs[i])+", Actual Output: "+str(output))

    return W_2, W_3



'''
PARAMETERS: list[] (testing_data), list[] (testing_outputs), list [[]] (W_2 - weight matrix 2), list [[]] (W_2 - weight matrix 3)
RETURN: N/A

DESCRIPTION: This function tests the neural network by using the latest updated weight matrices and sending in brand new datasets as inputs that the 
network hasn't seen before, prints out each input as well as the corresponding expected output, and the network's actual output/prediction. This shows 
off how accurate the network is with inputs it has never seen before.
'''
def testing(testing_data,testing_outputs,W_2,W_3):
    print("----------------------------")

    # iterate through all testing datasets and print them out along with their corresponding outputs (both expected and actual)
    for i in range(len(testing_data)):
        actual_output = forward_pass(testing_data[i],W_2,W_3)[0]
        print("Testing data: "+str(testing_data[i])+", Expected Output: "+str(testing_outputs[i]) + ", Actual Output: "+ str(actual_output))




# parameters: training( maximum epochs, training dataset, expected outputs, learning rate )
weight_matrices = training(MAX_EPOCH,training_data,expected_outputs,LEARNING_RATE)   

# # parameters: testing( testing dataset, testing outputs, W2, W3)
testing(testing_data,testing_outputs,weight_matrices[0],weight_matrices[1])


