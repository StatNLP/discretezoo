import csv
import numpy
import matplotlib.pyplot as plt
import math
import sys

output_name = sys.argv[1]
title = sys.argv[2]
paths = []
labels = []

for index, argument in enumerate(sys.argv[3:]):
    if (index % 2) == 0:
        labels.append(argument)
    else:
        paths.append(argument)

for label, path in zip(labels, paths):
    file = open(path)
    csv_file = csv.reader(file)
    csv_lines = list(csv_file)

    query_counts = []
    header = csv_lines[0]
    data = csv_lines[1:]
    total = 0

    for point in data:
        if point[-1] == "Successful":
            query_counts.append(math.ceil(float(point[1])))
        if point[-1] != "Skipped":
            total += 1

    count_dict = dict()
    count_dict[0] = 0

    for count in query_counts:
        count_dict[count] = count_dict.get(count, 0) + 1

    values = list(count_dict.keys())
    values.sort()
    cummulative_count = [0]

    for value in values:
        cummulative_count.append(cummulative_count[-1] + count_dict[value])
        
    del cummulative_count[0]

    rates = []
    for cc in cummulative_count:
        rates.append(cc / total)

    plt.plot(values, rates, label=label)

plt.ylim(0, 1.1)
plt.xlim(0, 10000)
plt.xlabel("Query Count")
plt.ylabel("Success Rate")
plt.title(title)
plt.legend()
plt.savefig(output_name)