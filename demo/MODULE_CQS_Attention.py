import MODULE_utils as utils
import MODULE_visualize as visualize
from random import choice
import numpy as np
from statistics import stdev, mean


class Scheduler:
    def __init__(self, N, W, interest_set = None) -> None:
        self.N = N
        self.W = W
        self.m_r = N / W

        if interest_set == None:
            # a random interest set
            self.interest_set = choice(utils.All_interst_sets(self.W, './Interest_Sets'))
        else:
            self.interest_set = interest_set
        self.m = len(self.interest_set)

    def gen_TG_Tk_map(self):
        TG_Tk_map = {}
        q, r = divmod(self.N, self.W)
        bound = self.W - r
        #There is a smarter way that record the start and end index ONLY
        for Wk_i in range(self.W):
            if Wk_i < bound:
                TG_Tk_map[Wk_i] = [ele for ele in range(Wk_i*q, Wk_i*q+q)]
            else:
                TG_Tk_map[Wk_i] = [ele for ele in range(Wk_i*q + Wk_i-bound, Wk_i*q + Wk_i-bound+q+1)]
        self.TG_Tk_map = TG_Tk_map

    def gen_CQS(self):
        CQS = []
        CQS.append(self.interest_set)
        for i in range (1, self.W):
            tmp_set = []
            for ele in self.interest_set:
                tmp = (ele + i) % self.W
                tmp_set.append(tmp)
            CQS.append(tmp_set)
        return CQS

    def gen_pair_lists_before_distillation(self, CQS):
        pair_lists_before_distill = []
        for Wk_i in range(self.W):
            all_perm = utils.all_pairs_strict_upper(CQS[Wk_i])
            pair_lists_before_distill.append(all_perm)
        return pair_lists_before_distill

    def self_disstillation(self, Wk_idx, undistilled_pair_list):
        distilled_pair_list = []
        all_diff = []

        for each_pair in undistilled_pair_list:
                diff1 = (each_pair[0] - each_pair[1]) % self.W
                if diff1 in all_diff:
                    continue
                all_diff.append(diff1)
                diff2 = (each_pair[1] - each_pair[0]) % self.W
                if diff1 == diff2:
                    if Wk_idx < self.W / 2: 
                        distilled_pair_list.append(each_pair)
                else:
                    all_diff.append(diff2)
                    distilled_pair_list.append(each_pair)
        return distilled_pair_list

    def gen_distilled_pair_lists(self, pair_lists_before_distill):
        # pair lists after distillation
        distilled_pair_lists = []
        for Wk_idx in range(self.W):
            distilled_pair_list = self.self_disstillation(Wk_idx, pair_lists_before_distill[Wk_idx])
            distilled_pair_lists.append(distilled_pair_list)
        return distilled_pair_lists

    def gen_Distilled_CQS(self, distilled_pair_lists):
        Distilled_CQS = []
        for pair_list in distilled_pair_lists:
            mtrlL_i_in_TG = set()
            for tk1, tk2 in pair_list:
                mtrlL_i_in_TG.add(tk1)
                mtrlL_i_in_TG.add(tk2)
            # sort is needed for demo purpose
            #Distilled_CQS.append(sorted(list(mtrlL_i_in_TG)))
            # otherwise, no need to sort
            Distilled_CQS.append(list(mtrlL_i_in_TG))
        return Distilled_CQS

    # token index retrieval
    def gen_MtrlL(self, Distilled_CQS):
        if self.W == self.N:
            self.MtrlL = Distilled_CQS
        else:
            MtrlL = []
            for distilled_quorum in Distilled_CQS:
                MtrlL_i = []
                for ele in distilled_quorum:
                    MtrlL_i += self.TG_Tk_map[ele]
                MtrlL.append(MtrlL_i.copy())
            self.MtrlL = MtrlL
    
    def gen_BL(self, Distilled_CQS, distilled_pair_lists):
        BL_global = []
        BL = []   

        for i in range(self.W):
            BL_i_global = []
            material_list_in_TG_for_ban_list = Distilled_CQS[i].copy()
            all_TG_pair_for_ban_list = set(utils.all_pairs_in_upper(material_list_in_TG_for_ban_list))
            all_TG_pair_for_ban_list = all_TG_pair_for_ban_list - set(distilled_pair_lists[i]) - {(i,i)}
            for TG1, TG2 in all_TG_pair_for_ban_list:
                BL_i_global += utils.all_inter_pairs(self.TG_Tk_map[TG1], self.TG_Tk_map[TG2], TG1 != TG2)
            BL_global.append(BL_i_global.copy())
            BL_i = []
            for tk1, tk2 in BL_i_global:
                BL_i.append((self.MtrlL[i].index(tk1),self.MtrlL[i].index(tk2)))
            BL.append(BL_i.copy())

        self.BL_global = BL_global
        self.BL = BL
    
    # ONLY for validation purpose in this study
    def gen_TL(self, distilled_pair_lists):
        if self.W == self.N:
            self.TL = utils.restore_all_n_square_pairs_from_upper_matrix(distilled_pair_lists)
        else:
            TL = []
            for Wk_i in range(self.W):
                each_task_list = []
                each_task_list += utils.all_inter_pairs(self.TG_Tk_map[Wk_i], self.TG_Tk_map[Wk_i])
                for TG1, TG2 in distilled_pair_lists[Wk_i]:
                    each_task_list += utils.all_inter_pairs(self.TG_Tk_map[TG1], self.TG_Tk_map[TG2], True)
                TL.append(each_task_list)           
            self.TL = TL

class Workers:
    def __init__(self, W) -> None:
        self.W = W
    
    def receive_from_Scheduler(self, Qis, Kis, Vis, BLis):
        self.Qis = Qis
        self.Kis = Kis
        self.Vis = Vis
        self.BLis = BLis
        

    def local_computation(self, display = False):
        Ois = []
        Sis = []
        if display:
            print('='*5, 'Workers', '='*5)
        for Wk_i in range(self.W):
            Pi = self.Qis[Wk_i] @ self.Kis[Wk_i].T
            for x,y in self.BLis[Wk_i]:
                Pi[x, y] = -np.inf
            Pi = np.exp(Pi)
            # row sum
            Si = np.sum(Pi, axis = 1)
            Oi = Pi @ self.Vis[Wk_i]
            
            if display:
                mTki, d = self.Qis[Wk_i].shape
                print(f'Worker {Wk_i}: mTki = {mTki}, d = {d}')
                print(f'Pi: {Pi.shape}\n{Pi}\n')
                print(f'Si.T: {Si.shape}\n{Si}\n') 
                print(f'Oi: {Oi.shape}\n{Oi}\n')
                
            Ois.append(Oi)
            Sis.append(Si)
        self.Ois = Ois
        self.Sis = Sis

class Tiler:
    def __init__(self, N, d) -> None:
        self.N = N
        self.d = d

    def receive_from_Scheduler(self, MtrlL):
        self.MtrlL = MtrlL
    
    def receive_from_Workers(self, Ois, Sis):
        self.Ois = Ois
        self.Sis = Sis

    def compute_final_attention(self, display = False):
        O = np.zeros((self.N, self.d))
        S = np.zeros(self.N)
        for Wk_i in range(len(self.MtrlL)):
            O[self.MtrlL[Wk_i], :] += self.Ois[Wk_i]
            S[self.MtrlL[Wk_i]] += self.Sis[Wk_i]
        if display:
            print('='*5, 'Tiler', '='*5)
            print('Putting local Oi and Si together:')
            print(f'O\n{O}\n')
            print(f'S.T\n{S}\n')
            print(f'Final Attention O = O/S\n{O / S[:, None]}\n')
        return O / S[:, None]
            
class CQS_Attention:
    def __init__(self, Q, K, V, W, interest_set = None) -> None:
        assert 2 < W < 112, 'Current only support W = 3, ..., 111.'
        self.Q = Q
        self.K = K
        self.V = V
        self.W = W
        self.interest_set = interest_set
        self.N, self.d = Q.shape 
        assert self.N >= W, 'Please make sure N >= W.'
        assert Q.shape == K.shape == V.shape, 'Please make sure Q, K, V are the same shape.'
        self.subsequence_lengths = []
        self.mW_ratio = None
        self.TL = None
        
    def workflow(self, display = False):
        scheduler = Scheduler(self.N, self.W, self.interest_set)
        workers = Workers(self.W)
        tiler = Tiler(self.N, self.d)

        # Scheduler flow
        scheduler.gen_TG_Tk_map()
        CQS = scheduler.gen_CQS()
        pair_lists_before_distill = scheduler.gen_pair_lists_before_distillation(CQS)
        distilled_pair_lists = scheduler.gen_distilled_pair_lists(pair_lists_before_distill)
        Distilled_CQS = scheduler.gen_Distilled_CQS(distilled_pair_lists)
        scheduler.gen_MtrlL(Distilled_CQS)
        scheduler.gen_BL(Distilled_CQS, distilled_pair_lists)

        # Gather subsequence length information
        for MtrlLi in scheduler.MtrlL:
            self.subsequence_lengths.append(len(MtrlLi))
        self.mW_ratio = scheduler.m / self.W
                                            
        # Construct Qi, Ki, Vi of W subsequences
        Qis = []
        Kis = []
        Vis = []

        for Wk_i in range(self.W):
            Qis.append(self.Q[scheduler.MtrlL[Wk_i],:])
            Kis.append(self.K[scheduler.MtrlL[Wk_i],:])
            Vis.append(self.V[scheduler.MtrlL[Wk_i],:])
        
        if display:
            print('='*5, 'Scheduler', '='*5)
            print(f'N = {self.N}, W = {self.W}, d = {self.d}\n')            
            print(f'Interest Set\n{scheduler.interest_set}\n')
            print(f'TG-Tk map\n{scheduler.TG_Tk_map}\n')            
            print(f'CQS\n{CQS}\n')            
            print(f'undistilled pair list\n{pair_lists_before_distill}\n')
            print(f'distilled pair list\n{distilled_pair_lists}\n')            
            print(f'distilled CQS\n{Distilled_CQS}\n')            
            print(f'MtrlL\n{scheduler.MtrlL}\n')            
            scheduler.gen_TL(distilled_pair_lists)
            print(f'Task lists (ONLY for validation purpose)\n{scheduler.TL}\n')
            print(f'Ban lists (global index)\n{scheduler.BL_global}\n')
            print(f'Ban lists (reindexed)\n{scheduler.BL}\n')

        # Worker flow
        workers.receive_from_Scheduler(Qis, Kis, Vis, scheduler.BL)
        workers.local_computation(display)

        # Tiler flow
        tiler.receive_from_Scheduler(scheduler.MtrlL)
        tiler.receive_from_Workers(workers.Ois, workers.Sis)
        self.O = tiler.compute_final_attention(display)
    
    def validate_computation_correctness(self):
        O0 = utils.attention_computation(self.Q, self.K, self.V)
        return np.allclose(O0,self.O)

    def memory_consumption_summary(self):
        print(f'FYI, approximation of subsequence length ratio is m / W: {self.mW_ratio}. Most actual ratios are lower, hence better.')
        max_len = max(self.subsequence_lengths)
        print(f'Longest subsequence: {max_len}, ratio to N: {max_len / self.N}')
        min_len = min(self.subsequence_lengths)
        print(f'Shortest subsequence: {min_len}, ratio to N: {min_len / self.N}')
        avg_len = mean(self.subsequence_lengths)
        stdv_len = stdev(self.subsequence_lengths)
        print(f'Average length: {avg_len}, ratio to N: {avg_len / self.N}\nStandard deviation: {stdv_len}\n')
        print('Subsequence length distribution')
        visualize.subsequence_length_hist(self.subsequence_lengths)
    
    def visualize_P_partition(self, show_grid = True, save_path = None):
        if self.TL == None:
            scheduler = Scheduler(self.N, self.W)
            scheduler.gen_TG_Tk_map()
            CQS = scheduler.gen_CQS()
            pair_lists_before_distill = scheduler.gen_pair_lists_before_distillation(CQS)
            distilled_pair_lists = scheduler.gen_distilled_pair_lists(pair_lists_before_distill)
            scheduler.gen_TL(distilled_pair_lists)
        visualize.P_partition(self.N, self.W, scheduler.TL, show_grid, save_path)
