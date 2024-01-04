from itertools import combinations
from scipy.special import softmax

def attention_computation(Q, K, V):
    return softmax(Q @ K.T, axis = 1) @ V

def restore_all_n_square_pairs_from_upper_matrix(list_of_lists_in):
    for i in range(len(list_of_lists_in)):
        lower_half = []
        for ii, jj in list_of_lists_in[i]:
            lower_half.append((jj, ii))
        list_of_lists_in[i].append((i,i))
        list_of_lists_in[i] += lower_half
    return list_of_lists_in

def all_pairs_in_upper(A):
    return list(combinations(A, 2)) + [(ele, ele) for ele in A]

def all_pairs_strict_upper(A):
    res = list(combinations(A, 2))
    for i in range(len(res)):
        ele1, ele2 = res[i]
        if ele1 > ele2:
            res[i] = (ele2, ele1)
    return res

def all_inter_pairs(A, B, both_pair = False):
    res = []
    for ele1 in sorted(A):
        for ele2 in sorted(B):
            res.append((ele1, ele2))  
            if both_pair:
                res.append((ele2, ele1))  
    return res

def All_interst_sets(W, CQS_folder = '../Interest_Sets'):
    all_interst_sets = []
    f = open(f'{CQS_folder}/{W}/{W}.txt','r')
    f_lines = f.readlines()
    for ele in f_lines:
        tmp2 = ele.strip('\n')
        tmp2 = tmp2.strip(' ')
        if len(tmp2) == 0:
            continue
        tmp3 = tmp2.split(' ')
        all_interst_sets.append([int(each_num) for each_num in tmp3])
    # W > 111 to be implemented
    return all_interst_sets
