import statistics
# list comprehension 优雅的算法
input_list = [0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 5, 8, 9, 0, 0, 0, 0, 2, 9, 8]

sublists = []
sublist = []

len = 3
    # hyperparameter len(assigned duration of each note and silence)

for num in input_list:
    if num != 0:
        sublist.append(num)
    elif sublist:
        templist = [statistics.median(sublist)] * len
        sublists.append([0] * len)
        sublists.append(templist)
        sublist = []

if sublist:
    templist_last = [statistics.median(sublist)] * len
    sublists.append([0] * len)
    sublists.append(templist_last)

sublists = [elem for row in sublists for elem in row]
    # filter out the silence, and use the median value of non-zeros, make it a new list
print(sublists)
# filter out the silence, and use the median value