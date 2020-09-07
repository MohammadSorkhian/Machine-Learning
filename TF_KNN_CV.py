import os
import numpy as np
import sys
from io import StringIO
import math
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# cmd command
cmd = [i for i in sys.argv[1:]]  # we start from 1 since we jusy get desire arguments not the file name
x_train = cmd[0]
y_train = cmd[1]


class Instance:
    def __init__(self, givenPosition, givenName, givenString):
        self.position = givenPosition
        self.eu_distance = 9999
        self.ma_distance = 9999
        self.name = givenName
        self.string = givenString

    def __repr__(self):
        return f"[P({self.position})-D(eu-ma)({round(self.eu_distance,2)} - {self.ma_distance}), {self.name}, {self.string}]"


# Extracting all input data and build an object for each of them
#f_input = open("x_train.txt", "r")
f_input = open(x_train, "r")
input_data = []
i = 0
for l in f_input:
    temp = l.split()
    input_data.append(Instance(i, temp[0], temp[1]))
    i += 1


# Extracting all ys
#f_y = open("y_train.txt", "r")
f_y = open(y_train, "r")
y = []
for x in range(len(f_y.readline().split())):
    y.append([])
text = f_y.read()
c = StringIO(text)
y = np.loadtxt(c, unpack=True)


# Split the input_data to 5 parts in Random
train_partitions = []
test_partitions = []
kf = KFold(n_splits=5, shuffle=True, random_state=None)
for train_index, test_index in kf.split(input_data):
    train_partitions.append(train_index)
    test_partitions.append(test_index)


# (NO-Return) function that finds difference of an test string with all instances string in the trainingSet
def find_eu_ma_distance_between_a_test_and_trainingset(a_test_index, array_train_indices):
    chars_a_test_string = [char for char in input_data[a_test_index].string]  # Fetch chars of a test string
    for xxx in array_train_indices:  # For finding diff of the unseen string with ALL of training strings
        char_a_training_string = [char for char in input_data[xxx].string]  # Fetch chars of a training string
        diff_temp = abs(len(chars_a_test_string) - len(char_a_training_string))
        char_counter_temp = 0
        if len(char_a_training_string) > len(chars_a_test_string):
            char_a_training_string = char_a_training_string[0:len(chars_a_test_string)]
        for char in char_a_training_string:  # For finding diff of the test string with ONE of the training string
            if char != chars_a_test_string[char_counter_temp]:
                diff_temp += 1
            char_counter_temp += 1
        input_data[xxx].eu_distance = math.sqrt(diff_temp)  # Calculate Euclidean distance
        input_data[xxx].ma_distance = diff_temp  # Calculate Manhattan distance


# (Return) function returns an array of Ks NN of a test data with specific K.
# Rerturned array has two arrays of Ks NN (one for each eu & ma).
def sort_train_data_based_eu_ma(train_objs):
    pos_sorted_train_eu_ma = []  # This arr has two arrs of sorted train set positions based on eu and ma distances

    def s_distance_eu(x):  # this function is required for sorting an array of objects
        return x.eu_distance
    temp = sorted(train_objs, key=s_distance_eu)  # Sorts objects of training_data according to their distance
    temp_pos = [x.position for x in temp]
    pos_sorted_train_eu_ma.append(temp_pos)

    def s_distance_ma(x):  # this function is required for sorting an array of objects
        return x.ma_distance
    temp = sorted(train_objs, key=s_distance_ma)  # Sorts objects of training_data according to their distance
    temp_pos = [x.position for x in temp]
    pos_sorted_train_eu_ma.append(temp_pos)
    return pos_sorted_train_eu_ma


# (Return) function returns an array of y` for a test instance with specific K.
# This array has two arrays of y`, one for each eu & ma.
def cal_eu_ma_y_for_a_k(sorted_train_eu_ma, k):
    ys_eu_ma = []  # Consist of two Ys one for eu and other for ma distance in regarding specific K
    for sort in sorted_train_eu_ma:
        y_temp = np.array(y[sort[0]])
        for x in range(1, k):
            y_temp += y[sort[x]]
        y_temp = np.divide(y_temp, k)
        ys_eu_ma.append(y_temp)
    return ys_eu_ma


def find_actual_y_a_test(test_index):
    y_test_actual = y[test_index]
    return y_test_actual


def cal_spearman_correlation(ys_eu_ma, actual_y):
    spearman_corr = []
    for x in ys_eu_ma:
        coef, p = spearmanr(x, actual_y)
        spearman_corr.append(round(coef, 4))
    return spearman_corr


def calculate_final_spearman_corr():
    final_spearman_table = np.array(partitions_table[0])
    for x in range(1, len(partitions_table)):
        final_spearman_table += partitions_table[x]
    final_spearman_table = np.divide(final_spearman_table, len(partitions_table))
    return final_spearman_table


def calculate_standard_deviation():
    table_std_deviation = []
    for k in range(len(partitions_table[0])):  # For iterating rows (different Ks)
        std_dev_eu_ma = []
        for d in range(len(partitions_table[0][0])):  # For iterating columns (different distances)
            mean = 0
            for x in partitions_table:  # For calculating average for specific k-distance
                mean += x[k][d] / len(partitions_table)
            std_dev = 0
            for x in partitions_table:
                std_dev += ((x[k][d] - mean) ** 2) / len(partitions_table)
            std_dev = math.sqrt(std_dev)
            std_dev_eu_ma.append(round(std_dev, 2))
        table_std_deviation.append(std_dev_eu_ma)
    return table_std_deviation


def create_output_txt(k_array, final_spearman, final_deviation):
    output_f = open("model_selection_table.txt", 'w')
    output_f = open("model_selection_table.txt", 'a')
    output_f.write("\tEuclidean Dis\tManhattan Dis\n")
    maximum_Sp_Corr = 0
    k_max_sp_corr = []
    distance = []
    for x in final_spearman:  # Finding the best model
        temp = max(x[0], x[1])
        if temp > maximum_Sp_Corr:
            maximum_Sp_Corr = temp
    #print(final_spearman)
    for i, x in enumerate(final_spearman):
        if x[0] == maximum_Sp_Corr:
            distance.append("Euclidean")
            k_max_sp_corr.append(k_array[i])
        if x[1] == maximum_Sp_Corr:
            distance.append("Manhattan")
            k_max_sp_corr.append(k_array[i])
    for i, k in enumerate(k_array):  # Writing the out put file
        output_f.write(f"K={k}\t{round(final_spearman[i][0], 4)}±{final_deviation[i][0]}\t{round(final_spearman[i][1], 4)}±{final_deviation[i][1]}\n")
    output_f.write(f"Model chosen: k={k_max_sp_corr[-1]},\t{distance[-1]} distance")


k_array = [3, 5, 7, 9, 11]  # enter in ascending order
partitions_table = []
for p in range(len(train_partitions)):
    train_indices = train_partitions[p]
    test_indices = test_partitions[p]
    train_objs = [input_data[x] for x in train_indices]  # An array of train objects for using in find_knn_for_the_test_data
    table_spearman_corr_for_a_partition = np.array([[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]])
    for x in test_indices:
        y_actual = find_actual_y_a_test(x)
        find_eu_ma_distance_between_a_test_and_trainingset(x, train_indices)
        pos_sorted_train_eu_ma = sort_train_data_based_eu_ma(train_objs)  # Contains two arrays of position of sorted train set based on eu & ma distances
        table_spearman_corr_for_a_test_instance = []  # This table consists of y`s of a test data for different Ks and Ds
        for k in k_array:
            ys_eu_ma_for_a_k = cal_eu_ma_y_for_a_k(pos_sorted_train_eu_ma, k)  # An array contains predicted ys of specific K and different distances
            spearman_corr_regarding_a_k = cal_spearman_correlation(ys_eu_ma_for_a_k, y_actual)
            table_spearman_corr_for_a_test_instance.append(spearman_corr_regarding_a_k)
        table_spearman_corr_for_a_partition += table_spearman_corr_for_a_test_instance
    table_spearman_corr_for_a_partition = np.divide(table_spearman_corr_for_a_partition, len(test_indices))
    partitions_table.append(table_spearman_corr_for_a_partition)
final_spearman = calculate_final_spearman_corr()
final_deviation = calculate_standard_deviation()
create_output_txt(k_array, final_spearman, final_deviation)


# Plotting the results
'''
eu_ks = []  # Separating data for two tables (Euclidean and Manhattan)
ma_ks = []
for i in range(len(k_array)):
    temp_eu = []
    temp_ma = []
    for x in partitions_table:
        temp_eu.append(round(x[i][0], 3))
        temp_ma.append(round(x[i][1], 3))
    eu_ks.append(temp_eu)
    ma_ks.append(temp_ma)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))  # Building a figure and two axes
plt.subplots_adjust(wspace=0.4)  # adding more white space between two axes
lables = [x for x in k_array]
axes[0].set_title('Euclidean distance')
axes[1].set_title('Manhattan distance')
boxprops=dict(facecolor="red")
bplot1 = axes[0].boxplot(eu_ks, labels=lables, vert=True, patch_artist=True)  # Euclidean
bplot2 = axes[1].boxplot(ma_ks, labels=lables, vert=True, patch_artist=True)  # Manhattan
colors = ['teal' , 'teal' , 'teal', 'teal', 'teal']  # # Coloring and Labeling
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
for ax in axes:
    ax.yaxis.grid(False)
    ax.set_xlabel('K')
    ax.set_ylabel('Spearman Corr')
plt.savefig("task1")
plt.show()
'''