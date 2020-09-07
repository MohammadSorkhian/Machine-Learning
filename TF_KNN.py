import os
import numpy as np
import sys
import math

cmd = [i for i in sys.argv[1:]]  # we start from 1 since we jusy get desire arguments not the file name
x_train = cmd[0]
y_train = cmd[1]
x_unseen = cmd[2]


class Instance:
    def __init__(self,givenPosition,givenName,givenString,):
        self.position=givenPosition
        self.distance=9999
        self.name=givenName
        self.string=givenString
    def __repr__(self):
        return (f"[P({self.position})-D({self.distance}) , N({self.name}) , {self.string}]")


# This Array is used for creating output text
ys_unseen_final = []

# Extracting training data
f_train = open(x_train, "r")  # f_train = open("X_train.txt", "r")
train_data = []
i = 0
for l in f_train:
    temp=l.split()
    train_data.append(Instance(i, temp[0], temp[1]))
    i += 1

# Extracting unseen data
f_unseen = open(x_unseen, "r")  # f_unseen = open("X_unseen.txt", "r")
unseen_data = []
i = 0
for l in f_unseen:
    temp = l.split()
    unseen_data.append(Instance(i, temp[0], temp[1]))
    i += 1

# (NO-Return) function that finds difference of an unseen string with all trainig strings
def find_diff_an_unseen_str_with_all_training_str(index_unseen):
    if index_unseen >= len(unseen_data):
        return print("The 'unseen_index' is not in the acceptable range")
    else:
        chars_an_unseen_string = [char for char in unseen_data[index_unseen].string]  # Fetch chars of a unseen string
        for xxx in range(0, len(train_data)):  # For finding diff of the unseen string with ALL of training strings
            char_a_training_string = [char for char in train_data[xxx].string]  # Fetch chars of a training string
            char_counter_temp = 0
            diff_temp = 0
            if len(char_a_training_string) > len(chars_an_unseen_string):
                diff_temp += abs(len(chars_an_unseen_string)-len(char_a_training_string))
                char_a_training_string = char_a_training_string[0:len(chars_an_unseen_string)]
            for char in char_a_training_string:  # For finding diff of the unseen string with ONE of training string
                if char != chars_an_unseen_string[char_counter_temp]:
                    diff_temp += 1
                char_counter_temp += 1
            train_data[xxx].distance = math.sqrt(diff_temp)


# (Return) function that finds and returns Ks nearest neighbours
def find_knn_of_an_unseen_data(k, index_unseen):
    find_diff_an_unseen_str_with_all_training_str(index_unseen)

    def s_distance(x):  # this function is required for sorting an array of objects
        return x.distance
    sorted_distances = sorted(train_data, key=s_distance)  # Sorts objects of training_data according to their distance
    knn_obj = sorted_distances[0:k]
    return knn_obj


# (No-Return) function that calculate y` for an unseen_data and make an array out of it
def find_y_for_an_unseen_data(k, index_unseen):
    knn_objs = find_knn_of_an_unseen_data(k, index_unseen)
    f_y = open(y_train, "r")  # f_y = open("y_train.txt", "r")
    ys_knn_objs = []
    y_temp = []
    for obj in knn_objs:  # Extract ys for k-NN
        pos = obj.position
        f_y.readline()
        for x in f_y:
            y_temp.append(float(x.split()[pos]))
        ys_knn_objs.append(y_temp)
        y_temp = []
        f_y.seek(0)
    np_ys_knn_objs = np.array(ys_knn_objs)  # Convert array of kNNs ys to NumPy array for doing calcilation
    np_y_unseen_temp = np_ys_knn_objs[0]
    for number in range(1, len(np_ys_knn_objs)):
        np_y_unseen_temp += np_ys_knn_objs[number]
    np_y_unseen_temp_mean = np.divide(np_y_unseen_temp, k)
    np_y_unseen_temp_mean = [round(x, 4) for x in np_y_unseen_temp_mean]
    ys_unseen_final.append(np_y_unseen_temp_mean)


def create_output_text_for_ys_unseen_final():
    f_ys_unseen_final=open("Y_unseen.txt","w")
    f_ys_unseen_final=open("Y_unseen.txt","a")
    names_unseen_data=[x.name for x in unseen_data]
    for x in names_unseen_data:
        f_ys_unseen_final.write(x+"\t")
    f_ys_unseen_final.write("\n")
    for row in range(0, len(ys_unseen_final[0])):
        for x in range(0, len(ys_unseen_final)):
            f_ys_unseen_final.write(str(ys_unseen_final[int(x)][int(row)])+"\t")
        f_ys_unseen_final.write("\n")


for num in range(0, len(unseen_data)):
    find_y_for_an_unseen_data(6, num)
create_output_text_for_ys_unseen_final()

