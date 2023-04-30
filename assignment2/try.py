import statistics
# list comprehension 优雅的算法
input_list = [0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 5, 8, 9, 0, 0, 0, 0, 2, 9, 8]

sublists = []
sublist = []

for num in input_list:
    if num != 0:
        sublist.append(num)
    elif sublist:
        templist = [statistics.median(sublist)] * len(sublist)
        sublists.append(templist)
        sublist = []

if sublist:
    templist_last = [statistics.median(sublist)] * len(sublist)
    sublists.append(templist_last)

sublists = [elem for row in sublists for elem in row]
print(sublists)
# filter out the silence, and use the median value