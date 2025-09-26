import torch
import torch.distributed

from .redistribute import gather_from_group, split_to_group


class CommGroup(object):
    def __init__(self, ranks):
        assert isinstance(ranks, list) or isinstance(
            ranks, range
        ), "Rank list or range should be provided to create a CommGroup!"
        self.ranks = sorted(list(set(list(ranks))))
        self.size = len(self.ranks)
        self.group = torch.distributed.new_group(self.ranks)

    def has_rank(self, rank):
        if rank in self.ranks:
            self.intra_group_id = self.ranks.index(rank)
            return True
        return False

    # def allgather(self, input, is_input):
    #     return gather_from_group(input, self.group, is_input)

    # def split(self, input, is_input):
    #     return split_to_group(input, self.group, is_input)

    def print(self):
        print(self.ranks, end=" ")


def show_groups(groups):
    for group in groups:
        if group is None:
            print("None", end=" ")
        else:
            group.print()
    print()


def sort_ranks(ranks=None):
    if ranks is None:
        return None
    else:
        return sorted(list(set(list(ranks))))


def get_world_size(world_ranks=None):
    if world_ranks is None:
        return torch.distributed.get_world_size()
    else:
        assert isinstance(world_ranks, list)
        return len(world_ranks)

def get_group_rank(world_ranks=None):
    if world_ranks is None:
        return torch.distributed.get_rank()
    else:
        assert isinstance(world_ranks, list)
        return world_ranks.index(torch.distributed.get_rank())

def index_ranks(ranks, world_ranks=None):
    if world_ranks is None:
        return ranks
    else:
        assert isinstance(world_ranks, list)
        return [world_ranks[i] for i in ranks]


#consecutive must be True for tp
def gen_tp_group_dist(tp_size, pp_size, to_print=True, consecutive=True, world_ranks=None):
    world_ranks = sort_ranks(world_ranks)
    rank, world_size = torch.distributed.get_rank(), get_world_size(world_ranks)
    all_tp_groups, tp_group = [], None

    num_tp_groups = world_size // tp_size
    
    if consecutive:
        for i in range(num_tp_groups):
            ranks = range(i * tp_size, (i + 1) * tp_size)
            ranks = index_ranks(ranks, world_ranks)
            group = CommGroup(ranks)
            all_tp_groups.append(group)
            if group.has_rank(rank):
                tp_group = group
    
    if rank == 0 and to_print:
        print("TP groups:", end=" ")
        show_groups(all_tp_groups)
    return tp_group

#Consecutive must be false for cp
#mul_size = tp_size * sp_size
def gen_cp_group_dist(mul_size, cp_size, pp_size, to_print=True, consecutive=False, world_ranks=None):
    world_ranks = sort_ranks(world_ranks)
    rank, world_size = torch.distributed.get_rank(), get_world_size(world_ranks)
    all_cp_groups, cp_group = [], None
    dp_size = world_size // mul_size // pp_size // cp_size
    num_pp_groups = world_size // pp_size #Ranks per pipeline stage 
    
    #Assumes PP->TP/SP->CP->DP
    if not consecutive:
        for i in range(pp_size):
            pp_base_rank = i * num_pp_groups
            for j in range(dp_size):
                dp_base_rank = pp_base_rank + j * (mul_size * cp_size) #DP stride: mul * cp
                for k in range(mul_size): 
                    base_rank = dp_base_rank + k
                    ranks = list(range(base_rank, base_rank + mul_size * cp_size, mul_size))
                    ranks = index_ranks(ranks, world_ranks)
                    group = CommGroup(ranks)
                    all_cp_groups.append(group)
                    if group.has_rank(rank):
                        cp_group = group
    if rank == 0 and to_print:
        print("CP groups:", end=" ")
        show_groups(all_cp_groups)
    return cp_group

#Consecutive must be false for dp
def gen_dp_group_dist(mul_size, cp_size, pp_size, to_print=True, consecutive=False, world_ranks=None):
    world_ranks = sort_ranks(world_ranks)
    rank, world_size = torch.distributed.get_rank(), get_world_size(world_ranks)
    all_dp_groups, dp_group = [], None
    num_pp_groups = world_size // pp_size
    dp_stride = mul_size * cp_size

    if not consecutive:
        # Logic assumes PP -> TP -> CP -> DP ordering
        for i in range(pp_size): # Iterate PP stages
            pp_base_rank = i * num_pp_groups
            for j in range(mul_size * cp_size): # Iterate combined (TP*SP, CP) indices within a DP block
                # This index 'j' represents a unique (tp, cp) combination within the DP group
                start_rank_for_group = pp_base_rank + j
                ranks = list(range(start_rank_for_group, pp_base_rank + num_pp_groups, dp_stride))
                ranks = index_ranks(ranks, world_ranks)
                group = CommGroup(ranks)
                if group.has_rank(rank):
                    dp_group = group
    if rank == 0 and to_print:
        print("DP groups:", end=" ")
        show_groups(all_dp_groups)
    return dp_group

#consecutive must be True for sp
def gen_sp_group_dist(sp_size, pp_size, to_print=True, consecutive=True, world_ranks=None):
    world_ranks = sort_ranks(world_ranks)
    rank, world_size = torch.distributed.get_rank(), get_world_size(world_ranks)
    all_sp_groups, sp_group = [], None
    # dp_size = world_size // sp_size // pp_size
    # num_pp_groups = world_size // pp_size
    num_sp_groups = world_size // sp_size

    if consecutive:
        for i in range(num_sp_groups):
            ranks = range(i * sp_size, (i + 1) * sp_size)
            ranks = index_ranks(ranks, world_ranks)
            group = CommGroup(ranks)
            all_sp_groups.append(group)
            if group.has_rank(rank):
                sp_group = group
    # else:
    #     for i in range(pp_size):
    #         start_rank = i * num_pp_groups
    #         end_rank = (i + 1) * num_pp_groups
    #         for j in range(dp_size):
    #             ranks = range(start_rank + j, end_rank, dp_size)
    #             ranks = index_ranks(ranks, world_ranks)
    #             group = CommGroup(ranks)
    #             all_sp_groups.append(group)
    #             if group.has_rank(rank):
    #                 sp_group = group

    if rank == 0 and to_print:
        print("SP groups:", end=" ")
        show_groups(all_sp_groups)
    return sp_group


def gen_pp_group_dist(pp_size, to_print=True, world_ranks=None):
    world_ranks = sort_ranks(world_ranks)
    rank, world_size = torch.distributed.get_rank(), get_world_size(world_ranks)
    all_pp_groups, pp_group = [], None
    num_pp_groups = world_size // pp_size
    for i in range(num_pp_groups):
        ranks = range(i, world_size, num_pp_groups)
        ranks = index_ranks(ranks, world_ranks)
        group = CommGroup(ranks)
        all_pp_groups.append(group)
        if group.has_rank(rank):
            pp_group = group

    if rank == 0 and to_print:
        print("PP groups:", end=" ")
        show_groups(all_pp_groups)
    return pp_group, all_pp_groups


def gen_embedding_group_dist(pp_size, all_pp_groups, to_print=True):
    rank = torch.distributed.get_rank()
    all_embedding_groups, embedding_group = [], None
    for pp_group in all_pp_groups:
        if pp_size > 1:
            embedding_ranks = [pp_group.ranks[0], pp_group.ranks[-1]]
        else:
            embedding_ranks = [pp_group.ranks[0]]
        group = CommGroup(embedding_ranks)
        all_embedding_groups.append(group)
        if group.has_rank(rank):
            embedding_group = group

    if rank == 0 and to_print:
        print("Embedding groups:", end=" ")
        show_groups(all_embedding_groups)
    return embedding_group

#sequence group = megatron sp group(tp group) + cp group or ulysses sp group + cp group
def gen_sep_group_dist(tp_size, cp_size, pp_size, to_print=True, consecutive=True, world_ranks=None):
    world_ranks = sort_ranks(world_ranks)
    rank, world_size = torch.distributed.get_rank(), get_world_size(world_ranks)
    all_sep_groups, sep_group = [], None
    num_sep_groups = world_size // (tp_size * cp_size)
   
    if consecutive:
        for i in range(num_sep_groups):
            ranks = range(i * tp_size * cp_size, (i + 1) * tp_size * cp_size)
            ranks = index_ranks(ranks, world_ranks)
            group = CommGroup(ranks)
            all_sep_groups.append(group)
            if group.has_rank(rank):
                sep_group = group

    if rank == 0 and to_print:
        print("SEP groups:", end=" ")
        show_groups(all_sep_groups)
    return sep_group

def get_tp_group_dict_dist(all_tp_sizes, pp_size, consecutive=True, world_ranks=None):
    tp_sizes_set = list(set(all_tp_sizes))
    tp_group_dict = {}
    for tp_size in tp_sizes_set:
        tp_group_dict[tp_size] = gen_tp_group_dist(
            tp_size, pp_size, to_print=False, consecutive=consecutive, world_ranks=world_ranks
        )
    return tp_group_dict

def get_cp_group_dict_dist(all_tp_sizes, all_sp_sizes, all_cp_sizes, pp_size, consecutive=False, world_ranks=None):
    all_mul_sizes = []
    world_size = torch.distributed.get_world_size() if world_ranks is None else torch.distributed.get_world_size(world_ranks)
    for tp_size, sp_size in zip(all_tp_sizes, all_sp_sizes):
        all_mul_sizes.append(tp_size * sp_size)
    mul_sizes_set = list(set(all_mul_sizes))
    cp_group_dict = {}
    for mul_size in mul_sizes_set:
        cp_group_dict[mul_size] = {}
        for cp_size in all_cp_sizes:
            if mul_size * cp_size <= world_size:
                cp_group_dict[mul_size][cp_size] = gen_cp_group_dist(
                    mul_size, cp_size, pp_size, to_print=False, consecutive=consecutive, world_ranks=world_ranks
                )
    return cp_group_dict

def get_dp_group_dict_dist(all_tp_sizes, all_sp_sizes, all_cp_sizes, pp_size, consecutive=False, world_ranks=None):
    all_mul_sizes = []
    world_size = torch.distributed.get_world_size() if world_ranks is None else torch.distributed.get_world_size(world_ranks)
    world_size_per_stage = world_size // pp_size
    for tp_size, sp_size in zip(all_tp_sizes, all_sp_sizes):
        all_mul_sizes.append(tp_size * sp_size)

    mul_sizes_set = list(set(all_mul_sizes))
    dp_group_dict = {}
    for mul_size in mul_sizes_set:
        dp_group_dict[mul_size] = {}
        for cp_size in all_cp_sizes:
            if mul_size * cp_size <= world_size_per_stage:
                dp_group_dict[mul_size][cp_size] = gen_dp_group_dist(
                    mul_size, cp_size, pp_size, to_print=False, consecutive=consecutive, world_ranks=world_ranks
                )
    return dp_group_dict

def get_sp_group_dict_dist(all_sp_sizes, pp_size, consecutive=True, world_ranks=None):
    sp_sizes_set = list(set(all_sp_sizes))
    sp_group_dict = {}
    for sp_size in sp_sizes_set:
        sp_group_dict[sp_size] = gen_sp_group_dist(
            sp_size, pp_size, to_print=False, consecutive=consecutive, world_ranks=world_ranks
        )
    return sp_group_dict

# def gen_redistributed_group(tp_size_old, tp_size_new, tp_consec_old, tp_consec_new, tp_group_old, tp_group_new):
#     if tp_size_old == tp_size_new and tp_consec_old == tp_consec_new:
#         return (None, None)
#     tp_group_old = None if tp_size_old == 1 else tp_group_old
#     tp_group_new = None if tp_size_new == 1 else tp_group_new
#     return (tp_group_old, tp_group_new)

#when we use ulysses sp,the tp size will be changed into sp size
def gen_redistributed_group_with_cp(tp_size_old, tp_size_new, tp_consec_old, tp_consec_new, tp_group_old, tp_group_new,
                                    cp_size_old, cp_size_new, cp_consec_old, cp_consec_new, cp_group_old, cp_group_new, ):
    if tp_size_old == tp_size_new and tp_consec_old == tp_consec_new and cp_size_old == cp_size_new and cp_consec_old == cp_consec_new:
        return (None, None, None, None)
    tp_group_old = None if tp_size_old == 1 else tp_group_old
    tp_group_new = None if tp_size_new == 1 else tp_group_new
    cp_group_old = None if cp_size_old == 1 else cp_group_old
    cp_group_new = None if cp_size_new == 1 else cp_group_new
    return (tp_group_old, tp_group_new, cp_group_new, cp_group_old)

# def gen_redistributed_group_with_cp(tp_size_old, tp_size_new, cp_size_old, cp_size_new, tp_group_old, tp_group_new, cp_group_old, cp_group_new):
#     if tp_size_old * cp_size_old == tp_size_new * cp_size_new:
#         return (None, None)
#     split_group = None if tp_size_old * cp_size_old == 1 else merge_tp_cp_groups(tp_group_old, cp_group_old)
#     allgather_group = None if tp_size_new * cp_size_new == 1 else merge_tp_cp_groups(tp_group_new, cp_group_new)
#     return (split_group, allgather_group)

def merge_redistributed_group(split_tp_sp_cp_group, allgather_tp_sp_cp_group, world_ranks=None):
    if split_tp_sp_cp_group is None or allgather_tp_sp_cp_group is None:  
        return None, None

    split_tp_sp_cp_size = split_tp_sp_cp_group.size
    allgather_tp_sp_cp_size = allgather_tp_sp_cp_group.size
    # if split_tp_sp_cp_group.ranks[1] - split_tp_sp_cp_group.ranks[0] == 1:
    #     split_consecutive = 1
    # else:
    #     split_consecutive = 0
    # if allgather_tp_sp_cp_group.ranks[1] - allgather_tp_sp_cp_group.ranks[0] == 1:
    #     allgather_consecutive = 1
    # else:
    #     allgather_consecutive = 0

    # if split_consecutive == 0 or allgather_consecutive == 0:
    #     return None, None

    world_ranks = sort_ranks(world_ranks)
    rank, world_size = torch.distributed.get_rank(), get_world_size(world_ranks)

    if split_tp_sp_cp_size > allgather_tp_sp_cp_size:
        num_tp_sp_cp_groups = world_size // split_tp_sp_cp_size
        # mul = split_tp_sp_cp_size // allgather_tp_sp_cp_size
        for i in range(num_tp_sp_cp_groups):
            for j in range(allgather_tp_sp_cp_size):
                ranks = range(i * split_tp_sp_cp_size + j, (i + 1) * split_tp_sp_cp_size + j, allgather_tp_sp_cp_size)
                group = CommGroup(ranks)
                if group.has_rank(rank):
                    fused_group = group
        return fused_group, None

    if split_tp_sp_cp_size < allgather_tp_sp_cp_size:
        num_tp_sp_cp_groups = world_size // allgather_tp_sp_cp_size
        # mul = allgather_tp_sp_cp_size // split_tp_sp_cp_size
        for i in range(num_tp_sp_cp_groups):
            for j in range(split_tp_sp_cp_size):
                ranks = range(i * allgather_tp_sp_cp_size + j, (i + 1) * allgather_tp_sp_cp_size + j, split_tp_sp_cp_size)
                group = CommGroup(ranks)
                if group.has_rank(rank):
                    fused_group = group

        return None, fused_group
    if split_tp_sp_cp_size == allgather_tp_sp_cp_size:
        return None, None
    assert False, "merge_redistributed_group error!"


# def gen_seq_data_group_dist(pp_size, to_print, world_ranks=None):
#     world_ranks = sort_ranks(world_ranks)
#     rank, world_size = torch.distributed.get_rank(), get_world_size(world_ranks)
#     all_seq_data_groups, seq_data_group = [], None
#     seq_data_world_size = world_size // pp_size
#     for i in range(pp_size):
#         ranks = range(i * seq_data_world_size, (i + 1) * seq_data_world_size)
#         ranks = index_ranks(ranks, world_ranks)
#         group = CommGroup(ranks)
#         all_seq_data_groups.append(group)
#         if group.has_rank(rank):
#             seq_data_group = group

#     # if rank == 0 and to_print:
#     #     print("seq_data groups:", end = ' ')
#     #     show_groups(all_seq_data_groups)
#     return seq_data_group

#For FSDP Groups
def gen_seq_data_group_dist(pp_size, tp_size, to_print, world_ranks=None):
    world_ranks = sort_ranks(world_ranks)
    rank, world_size = torch.distributed.get_rank(), get_world_size(world_ranks)
    all_seq_data_groups, seq_data_group = [], None
    if tp_size == 1:
        seq_data_world_size = world_size // pp_size
        for i in range(pp_size):
            ranks = range(i * seq_data_world_size, (i + 1) * seq_data_world_size)
            ranks = index_ranks(ranks, world_ranks)
            group = CommGroup(ranks)
            all_seq_data_groups.append(group)
            if group.has_rank(rank):
                seq_data_group = group
    else:
        fsdp_tp_world_size = world_size // pp_size
        for i in range(pp_size):
            for j in range(tp_size):
                start = i * fsdp_tp_world_size + j
                ranks =  list(range(start, start + fsdp_tp_world_size, tp_size))
                ranks = index_ranks(ranks, world_ranks)
                group = CommGroup(ranks)
                all_seq_data_groups.append(group)
                if group.has_rank(rank):
                    seq_data_group = group
    # if rank == 0 and to_print:
    #     print("seq_data groups:", end = ' ')
    #     show_groups(all_seq_data_groups)
    return seq_data_group

# def merge_groups(tp_sp_group, cp_group):
#     merged_ranks = tp_sp_group.ranks + cp_group.ranks
#     merged_group = CommGroup(merged_ranks)
#     return merged_group

def gen_comm_groups(all_tp_sizes, all_sp_sizes, all_cp_sizes, pp_size, tp_consecutive_flags, show_rank=-1, world_ranks=None):
    world_ranks = sort_ranks(world_ranks)
    world_size = get_world_size(world_ranks)
    world_size_per_stage = world_size // pp_size
    for i in range(len(all_tp_sizes)):
        assert (
            all_tp_sizes[i] == 1 or all_sp_sizes[i] == 1
        ), "DeepSpeed Ulysses is not compatible with Megatron Tensor Parallel!"
    for i in range(len(all_tp_sizes)):
        tp_consec = tp_consecutive_flags[i]
        assert tp_consec == 0 or tp_consec == 1
        if all_tp_sizes[i] in [1, world_size_per_stage]:
            tp_consecutive_flags[i] = 1
    tp_groups, sp_groups, cp_groups, dp_groups = [], [], [], []
    #for input relocation
    #allgather_groups, split_groups = [None], [None]
    allgather_tp_sp_groups, split_tp_sp_groups = [None], [None]
    allgather_cp_groups, split_cp_groups = [None], [None]
    allgather_tp_sp_cp_groups, split_tp_sp_cp_groups = [None], [None]
    fused_split_groups, fused_allgather_groups = [None], [None]

    pp_group, all_pp_groups = gen_pp_group_dist(pp_size, to_print=False, world_ranks=world_ranks)
    embedding_group = gen_embedding_group_dist(pp_size, all_pp_groups, to_print=False)

    tp_group_dict, dp_group_dict, sp_group_dict, cp_group_dict = {}, {}, {}, {}
    for consec in [0, 1]: 
        tp_group_dict[consec] = get_tp_group_dict_dist(all_tp_sizes, pp_size, consec, world_ranks=world_ranks)
        dp_group_dict[consec] = get_dp_group_dict_dist(all_tp_sizes, all_sp_sizes, all_cp_sizes, pp_size, consec, world_ranks=world_ranks)
        cp_group_dict[consec] = get_cp_group_dict_dist(all_tp_sizes, all_sp_sizes, all_cp_sizes, pp_size, consec, world_ranks=world_ranks )
        sp_group_dict[consec] = get_sp_group_dict_dist(all_sp_sizes, pp_size, consec, world_ranks=world_ranks)  
    for i in range(len(all_tp_sizes)):
        tp_groups.append(tp_group_dict[tp_consecutive_flags[i]][all_tp_sizes[i]])
        cp_groups.append(cp_group_dict[1-tp_consecutive_flags[i]][all_tp_sizes[i] * all_sp_sizes[i]][all_cp_sizes[i]])
        dp_groups.append(dp_group_dict[1-tp_consecutive_flags[i]][all_tp_sizes[i] * all_sp_sizes[i]][all_cp_sizes[i]])
        sp_groups.append(sp_group_dict[tp_consecutive_flags[i]][all_sp_sizes[i]])
    vtp_data_group = dp_groups[0]#TODO:this code may be modified in future

    for i in range(1, len(all_tp_sizes)):
        if all_tp_sizes[i - 1] != 1:
            old_tp_size = all_tp_sizes[i - 1]
            old_tp_groups = tp_groups[i - 1]
        else:
            old_tp_size = all_sp_sizes[i - 1]
            old_tp_groups = sp_groups[i - 1]

        old_cp_size = all_cp_sizes[i - 1]
        old_cp_groups = cp_groups[i - 1]

        if all_tp_sizes[i] != 1:
            new_tp_size = all_tp_sizes[i]
            new_tp_groups = tp_groups[i]
        else:
            new_tp_size = all_sp_sizes[i]
            new_tp_groups = sp_groups[i]
        
        new_cp_size = all_cp_sizes[i]
        new_cp_groups = cp_groups[i]


        # split_group, allgather_group = gen_redistributed_group(
        #     old_tp_size, new_tp_size, tp_consecutive_flags[i - 1], tp_consecutive_flags[i], old_tp_groups, new_tp_groups
        # )

        # fused_split_group, fused_allgather_group = merge_redistributed_group(
        #     split_group, allgather_group, world_ranks=world_ranks
        # )

        split_tp_sp_group, allgather_tp_sp_group, split_cp_group, allgather_cp_group = gen_redistributed_group_with_cp(
            old_tp_size, new_tp_size, tp_consecutive_flags[i - 1], tp_consecutive_flags[i], old_tp_groups, new_tp_groups, 
            old_cp_size, new_cp_size, False, False, old_cp_groups, new_cp_groups)
        if old_tp_size == new_tp_size and old_cp_size == new_cp_size:
            split_tp_sp_cp_group = None
            allgather_tp_sp_cp_group = None
        else:
            split_tp_sp_cp_group = gen_sep_group_dist(old_tp_size, old_cp_size, pp_size, to_print=False, consecutive=True, world_ranks=world_ranks)
            allgather_tp_sp_cp_group = gen_sep_group_dist(new_tp_size, new_cp_size, pp_size, to_print=False, consecutive=True, world_ranks=world_ranks)
        
        fused_split_group, fused_allgather_group = merge_redistributed_group(
            split_tp_sp_cp_group, allgather_tp_sp_cp_group, world_ranks=world_ranks
        )

        # allgather_groups.append(allgather_group)
        # split_groups.append(split_group)

        allgather_tp_sp_groups.append(allgather_tp_sp_group)
        split_tp_sp_groups.append(split_tp_sp_group)

        allgather_cp_groups.append(allgather_cp_group)
        split_cp_groups.append(split_cp_group)

        allgather_tp_sp_cp_groups.append(allgather_tp_sp_cp_group)
        split_tp_sp_cp_groups.append(split_tp_sp_cp_group)

        fused_split_groups.append(fused_split_group)
        fused_allgather_groups.append(fused_allgather_group)
    
    seq_data_groups = [gen_seq_data_group_dist(pp_size, all_tp_sizes[i], to_print=True, world_ranks=world_ranks) for i in range(len(all_tp_sizes))]
    #seq_data_group = gen_seq_data_group_dist(pp_size, to_print=True, world_ranks=world_ranks)
    #seq_data_groups = [seq_data_group if all_tp_sizes[i] == 1 else dp_groups[i] for i in range(len(all_tp_sizes))]

    show_rank = 0
    if show_rank >= 0 and torch.distributed.get_rank() == show_rank:
        print("====================== Galvatron Communication Group ===========================")
        print("Embedding group for rank %d:" % show_rank)
        show_groups([embedding_group])
        print("TP groups for rank %d (all layers):" % show_rank)
        show_groups(tp_groups)
        print("SP groups for rank %d (all layers):" % show_rank)
        show_groups(sp_groups)
        print("CP groups for rank %d (all layers):" % show_rank)
        show_groups(cp_groups)
        print("DP groups for rank %d (all layers):" % show_rank)
        show_groups(dp_groups)
        print("SDP groups for rank %d (all layers):" % show_rank)
        show_groups(seq_data_groups)
        # print("Split groups for rank %d:" % show_rank)
        # show_groups(split_groups)
        print("Split TP/SP groups for rank %d:" % show_rank)
        show_groups(split_tp_sp_groups)
        print("AllGather TP/SP groups for rank %d:" % show_rank)
        show_groups(allgather_tp_sp_groups)
        print("Split CP groups for rank %d:" % show_rank)
        show_groups(split_cp_groups)
        print("AllGather CP groups for rank %d:" % show_rank)
        show_groups(allgather_cp_groups)
        print("Split TP/SP/CP groups for rank %d:" % show_rank)
        show_groups(split_tp_sp_cp_groups)
        print("AllGather TP/SP/CP groups for rank %d:" % show_rank)
        show_groups(allgather_tp_sp_cp_groups)
        print("Fused split groups for rank %d:" % show_rank)
        show_groups(fused_split_groups)
        print("Fused allgather groups for rank %d:" % show_rank)
        show_groups(fused_allgather_groups)
        print("================================================================================")
    return (
        pp_group,
        tp_groups,
        sp_groups,
        cp_groups,
        dp_groups,
        seq_data_groups,
        # allgather_groups,
        # split_groups,
        allgather_tp_sp_groups,
        split_tp_sp_groups,
        allgather_cp_groups,
        split_cp_groups,
        allgather_tp_sp_cp_groups,
        split_tp_sp_cp_groups,
        fused_allgather_groups,
        fused_split_groups,
        embedding_group,
        vtp_data_group,
    )
