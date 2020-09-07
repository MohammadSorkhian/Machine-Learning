import os
import sys
import matplotlib.pyplot as plt

# cmd command
input_data_name = sys.argv[1]  # we start from 1 since we just get desire arguments not the file name


class Instance:
    def __init__(self, probability, category):
        self.probability = probability
        self.category = category

    def __repr__(self):
        return f"({self.probability}-{self.category})"


class Pairs:
    def __init__(self, precition, recall, threshold):
        self.p = precition
        self.r = recall
        self.th = threshold

    def __repr__(self):
        return f"({round(self.p,3)}-{round(self.r,3)}){round(th,3)}"


# Extracting all input data
f_input = open(input_data_name, "r")
input_data = []
for l in f_input:
    temp = l.split()
    input_data.append(Instance(float(temp[0]), int(temp[1])))


def sort(obj):
    return obj.probability
input_data = sorted(input_data, key = sort)


# Calculate Precision Recall
def cal_p_r(tp, fp, fn):
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return p, r


# finding pairs
pairs = []
for obj in input_data[1:]:
    tp = 0
    fp = 0
    fn = 0
    th = obj.probability
    for x in input_data:
        if x.probability >= th:
            if x.category == 1:
                tp += 1
            else:
                fp += 1
        else:
            if x.category == 1:
                fn += 1
    p, r = cal_p_r(tp, fp, fn)
    pairs.append(Pairs(p, r, th))


# Create output text
output_f = open("PR_table.txt", "w")
output_f = open("PR_table.txt", "a")
for x in pairs:
    output_f.write(f"{x.p}\t{x.r}\t{x.th}\n")

# Calculate AUPRC
area = 0
for i in range(0, (len(pairs)-1)):
    height = pairs[i].p
    width = abs(pairs[i+1].r - pairs[i].r)
    area += height * width


# Plot and save the Precision Recall curve
precision = [x.p for x in pairs]
recall = [x.r for x in pairs]
plt.plot(recall, precision)
plt.title("PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.text(0.9, 0.9, f"AUPRC\n{round(area, 5)}")
plt.savefig("PRC")


'''
# The following codes can be used for testing the performance of our program
# predefined function P-R
from sklearn.metrics import precision_recall_curve
y_scores = [x.probability for x in input_data]
y_true = [x.category for x in input_data]
precision2, recall2, thresholds2 = precision_recall_curve(y_true, y_scores)
plt.plot(recall2, precision2)
plt.savefig("Original")
plt.show()

# predefined function AUPRC
from sklearn.metrics import average_precision_score
w = average_precision_score(y_true, y_scores)
print(w)
'''