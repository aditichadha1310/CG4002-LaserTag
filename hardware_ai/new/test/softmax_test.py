from scipy.special import softmax

list1 = [1,3,2]
print(type(list1))
softmax_list1 = softmax(list1)
print(type(softmax_list1))
print(softmax_list1)
print(list1)