#include <vector>
#include <algorithm>
#include <numeric>
#include <queue>
#include <cmath>
#include <limits>
#include <random>
#include <set>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/* ---------- 工具 ---------- */
static inline int node_func(int dev, int node_num) { return dev / 8; }

/* ---------- 带宽函数 ---------- */
inline double bandwidth_function(int i, int j, double v_intra, double v_inter) {
    if (i == j) {
        return 1e9;  // self-communication
    } else if ((i >> 3) != (j >> 3)) { // 优化: 位运算替代除法
        return v_inter;  // inter-node
    } else {
        return v_intra;  // intra-node
    }
}

/* ---------- 1. allocate_expert_replicas_precise ---------- */
struct HeapItem {
    double load;
    int replicas;
    int expert_id;
    bool operator<(const HeapItem& other) const {
        return (load / replicas) < (other.load / other.replicas);
    }
};

std::vector<int> allocate_expert_replicas_precise(const std::vector<double>& expert_loads,
                                                  int device_num,
                                                  int capacity) {
    const int total_capacity = device_num * capacity;
    const int expert_num = expert_loads.size();
    const int node_num = device_num / 8;

    std::priority_queue<HeapItem> max_heap;
    int now_capacity;
    if (1LL * node_num * expert_num < total_capacity) {
        for (int i = 0; i < expert_num; ++i)
            max_heap.push({expert_loads[i], node_num, i});
        now_capacity = node_num * expert_num;
    } else {
        for (int i = 0; i < expert_num; ++i)
            max_heap.push({expert_loads[i], 1, i});
        now_capacity = expert_num;
    }

    std::vector<int> result(expert_num, 0);
    while (now_capacity < total_capacity) {
        HeapItem item = max_heap.top(); max_heap.pop();
        if (item.replicas >= node_num) {
            if (now_capacity + node_num <= total_capacity) {
                max_heap.push({item.load, item.replicas + node_num, item.expert_id});
                now_capacity += node_num;
            } else {
                result[item.expert_id] = item.replicas;
            }
        } else {
            max_heap.push({item.load, item.replicas + 1, item.expert_id});
            now_capacity += 1;
        }
    }
    while (!max_heap.empty()) {
        result[max_heap.top().expert_id] = max_heap.top().replicas;
        max_heap.pop();
    }
    return result;
}

/* ---------- 2. distribute_expert_load_precise ---------- */
std::vector<double> distribute_expert_load_precise(double load, int replicas) {
    return std::vector<double>(replicas, load / replicas);
}

/* ---------- 2a. generate_perturbed_replicas ---------- */
std::vector<std::vector<int>> generate_perturbed_replicas(
    const std::vector<int>& base_replicas,
    int device_num,
    int num_perturbations) {
    
    std::vector<std::vector<int>> result;
    result.push_back(base_replicas); // 包含原始方案
    
    const int n_expert = base_replicas.size();
    const int node_num = device_num / 8;
    const int total_replicas = std::accumulate(base_replicas.begin(), base_replicas.end(), 0);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> expert_dist(0, n_expert - 1);

    std::set<std::vector<int>> unique_replicas; // 用于去重
    unique_replicas.insert(base_replicas);
    
    for (int iter = 0; iter < num_perturbations && unique_replicas.size() < num_perturbations + 1; ++iter) {
        std::vector<int> new_replicas = base_replicas;
        
        // 随机选择两个不同的专家进行交换
        int from_expert = expert_dist(gen);
        int to_expert = expert_dist(gen);
        while (to_expert == from_expert) {
            to_expert = expert_dist(gen);
        }
        
        // 确定交换的副本数量
        int transfer_amount;
        if (new_replicas[from_expert] % node_num == 0 && new_replicas[to_expert] % node_num == 0) {
            transfer_amount = node_num;
        } else {
            transfer_amount = 1;
        }
        if (new_replicas[from_expert] - transfer_amount <= 0) continue;
        new_replicas[from_expert] -= transfer_amount;
        new_replicas[to_expert] += transfer_amount;
        
        if (unique_replicas.find(new_replicas) == unique_replicas.end()) {
            unique_replicas.insert(new_replicas);
            result.push_back(new_replicas);
        }
    }
    
    return result;
}

/* ---------- 3. get_greedy_placement ---------- */
struct GreedyResult {
    std::vector<std::vector<int>> A;   // [expert][device]
    double max_load;
};

GreedyResult get_greedy_placement(const std::vector<int>& expert_replicas,
                                  const std::vector<double>& expert_loads,
                                  int n_device,
                                  int n_expert,
                                  int C_e) {
    const int node_num = n_device / 8;

    std::vector<std::vector<int>> A(n_expert, std::vector<int>(n_device, 0));
    std::vector<int> device_expert_count(n_device, 0);
    std::vector<double> device_loads(n_device, 0.0);

    /* 生成所有 (expert, load) 并按 load 降序 */
    std::vector<std::pair<int, double>> expert_list;
    for (int expert = 0; expert < n_expert; ++expert) {
        int replicas = expert_replicas[expert];
        auto replica_loads = distribute_expert_load_precise(expert_loads[expert], replicas);
        for (double load : replica_loads)
            expert_list.emplace_back(expert, load);
    }
    std::sort(expert_list.begin(), expert_list.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    auto node_func_local = [node_num](int dev) { return dev / 8; };

    for (const auto& [expert, load] : expert_list) {
        std::vector<int> available;
        for (int i = 0; i < n_device; ++i)
            if (device_expert_count[i] < C_e) available.push_back(i);
        if (available.empty())
            throw std::runtime_error("No more capacity");

        std::vector<int> existing_nodes(node_num, 0);
        for (int dev = 0; dev < n_device; ++dev)
            if (A[expert][dev]) existing_nodes[node_func_local(dev)]+=A[expert][dev];

        int min_node_cnt = *std::min_element(existing_nodes.begin(), existing_nodes.end());
        std::vector<int> new_node_devs;
        for (int dev : available)
            if (existing_nodes[node_func_local(dev)] == min_node_cnt)
                new_node_devs.push_back(dev);

        int best_device;
        if (!new_node_devs.empty()) {
            best_device = *std::min_element(new_node_devs.begin(), new_node_devs.end(),
                                            [&](int a, int b) { return device_loads[a] < device_loads[b]; });
        } else {
            best_device = *std::min_element(available.begin(), available.end(),
                                            [&](int a, int b) { return device_loads[a] < device_loads[b]; });
        }

        A[expert][best_device] += 1;
        device_loads[best_device] += load;
        device_expert_count[best_device] += 1;
    }

    double max_load = *std::max_element(device_loads.begin(), device_loads.end());
    return {std::move(A), max_load};
}

/* ---------- 4. _generate_smart_routing ---------- */
py::array_t<double> generate_smart_routing(int n_device, int n_expert, 
                                           const py::array_t<double>& E_np,
                                           const py::array_t<double>& A_np,
                                           double v_intra = 100.0,
                                           double v_inter = 12.0) {
    /* 解析 numpy 数组 */
    py::buffer_info E_buf = E_np.request();
    py::buffer_info A_buf = A_np.request();
    
    if (E_buf.ndim != 2 || E_buf.shape[0] != n_device || E_buf.shape[1] != n_expert)
        throw std::runtime_error("E shape mismatch");
    if (A_buf.ndim != 2 || A_buf.shape[0] != n_expert || A_buf.shape[1] != n_device)
        throw std::runtime_error("A shape mismatch");

    const double* E_ptr = static_cast<double*>(E_buf.ptr);
    const double* A_ptr = static_cast<double*>(A_buf.ptr);
    
    /* 创建输出数组 S[n_device][n_expert][n_device] */
    std::vector<size_t> shape = {static_cast<size_t>(n_device), 
                                 static_cast<size_t>(n_expert), 
                                 static_cast<size_t>(n_device)};
    auto S_np = py::array_t<double>(shape);
    py::buffer_info S_buf = S_np.request();
    double* S_ptr = static_cast<double*>(S_buf.ptr);
    
    /* 初始化 S 为 0 */
    std::fill(S_ptr, S_ptr + n_device * n_expert * n_device, 0.0);
    
    const int gpus_per_node = 8;
    
    /* 构建每个专家的位置和权重 */
    std::vector<std::vector<int>> expert_locations(n_expert);
    std::vector<std::vector<int>> expert_weights(n_expert);
    
    for (int expert = 0; expert < n_expert; ++expert) {
        for (int device = 0; device < n_device; ++device) {
            if (A_ptr[expert * n_device + device] > 0) {
                expert_locations[expert].push_back(device);
                expert_weights[expert].push_back(static_cast<int>(A_ptr[expert * n_device + device]));
            }
        }
    }
    
    /* 处理每个源设备和专家 */
    for (int src_device = 0; src_device < n_device; ++src_device) {
        int src_node = src_device / gpus_per_node;
        
        for (int expert = 0; expert < n_expert; ++expert) {
            double tokens_for_expert = E_ptr[src_device * n_expert + expert];
            if (tokens_for_expert == 0) continue;
            
            if (expert_locations[expert].empty()) continue;
            
            /* 分离节点内和节点间位置 */
            std::vector<int> intra_locations, intra_weights;
            std::vector<int> inter_locations, inter_weights;
            
            for (size_t idx = 0; idx < expert_locations[expert].size(); ++idx) {
                int location = expert_locations[expert][idx];
                int weight = expert_weights[expert][idx];
                int target_node = location / gpus_per_node;
                
                if (target_node == src_node) {
                    if (weight != 0) {
                        intra_locations.push_back(location);
                        intra_weights.push_back(weight);
                    }
                } else {
                    if (weight != 0) {
                        inter_locations.push_back(location);
                        inter_weights.push_back(weight);
                    }
                }
            }
            
            double remaining_tokens = tokens_for_expert;
            
            /* 阶段1: 节点内路由 (均匀分布) */
            if (!intra_locations.empty()) {
                int intra_count = 0;
                for (int w : intra_weights) intra_count += w;
                
                double tokens_per_location = remaining_tokens / intra_count;
                double extra_tokens = std::fmod(remaining_tokens, intra_count);
                
                for (size_t idx = 0; idx < intra_locations.size(); ++idx) {
                    int location = intra_locations[idx];
                    int weight = intra_weights[idx];
                    
                    double tokens_to_assign = tokens_per_location * weight;
                    double extra = std::min(static_cast<double>(weight), extra_tokens);
                    tokens_to_assign += extra;
                    extra_tokens -= extra;
                    
                    S_ptr[src_device * n_expert * n_device + expert * n_device + location] += tokens_to_assign;
                }
                
                remaining_tokens = 0;
            }
            
            /* 阶段2: 节点间路由 (均匀分布剩余token) */
            if (remaining_tokens > 0 && !inter_locations.empty()) {
                int inter_count = 0;
                for (int w : inter_weights) inter_count += w;
                
                double tokens_per_location = remaining_tokens / inter_count;
                double extra_tokens = std::fmod(remaining_tokens, inter_count);
                
                for (size_t idx = 0; idx < inter_locations.size(); ++idx) {
                    int location = inter_locations[idx];
                    int weight = inter_weights[idx];
                    
                    double tokens_to_assign = tokens_per_location * weight;
                    double extra = std::min(static_cast<double>(weight), extra_tokens);
                    tokens_to_assign += extra;
                    extra_tokens -= extra;
                    
                    S_ptr[src_device * n_expert * n_device + expert * n_device + location] += tokens_to_assign;
                }
            }
        }
    }
    
    return S_np;
}

/* ---------- 5. generate_smart_routing_and_calculate_time ---------- */
std::pair<py::array_t<double>, double> generate_smart_routing_and_calculate_time(
    int n_device, int n_expert, 
    const py::array_t<double>& E_np,
    const py::array_t<double>& A_np,
    double M_token,
    double v_comp = 0.001484375, // ms
    double v_intra = 216.269, // GB/s
    double v_inter = 88.9539) {
    
    /* 解析 numpy 数组 */
    py::buffer_info E_buf = E_np.request();
    py::buffer_info A_buf = A_np.request();
    
    if (E_buf.ndim != 2 || E_buf.shape[0] != n_device || E_buf.shape[1] != n_expert)
        throw std::runtime_error("E shape mismatch");
    if (A_buf.ndim != 2 || A_buf.shape[0] != n_expert || A_buf.shape[1] != n_device)
        throw std::runtime_error("A shape mismatch");

    const double* E_ptr = static_cast<double*>(E_buf.ptr);
    const double* A_ptr = static_cast<double*>(A_buf.ptr);
    
    /* 创建输出数组 S[n_device][n_expert][n_device] */
    std::vector<size_t> shape = {static_cast<size_t>(n_device), 
                                 static_cast<size_t>(n_expert), 
                                 static_cast<size_t>(n_device)};
    auto S_np = py::array_t<double>(shape);
    py::buffer_info S_buf = S_np.request();
    double* S_ptr = static_cast<double*>(S_buf.ptr);
    
    /* 初始化 S 为 0 */
    std::fill(S_ptr, S_ptr + n_device * n_expert * n_device, 0.0);
    
    /* 初始化时间计算变量 */
    std::vector<double> comm_times(n_device, 0.0);
    std::vector<double> comp_times(n_device, 0.0);
    
    const int gpus_per_node = 8;
    
    /* 构建每个专家的位置和权重 - 使用优化的数据结构 */
    struct ExpertLocation {
        int device;
        int weight;
    };
    std::vector<std::vector<ExpertLocation>> expert_locations(n_expert);
    
    for (int expert = 0; expert < n_expert; ++expert) {
        for (int device = 0; device < n_device; ++device) {
            int weight = static_cast<int>(A_ptr[expert * n_device + device]);
            if (weight > 0) {
                expert_locations[expert].push_back({device, weight});
            }
        }
    }
    
    /* 处理每个源设备和专家 */
    for (int src_device = 0; src_device < n_device; ++src_device) {
        int src_node = src_device >> 3; // 优化: 位运算替代除法
        
        for (int expert = 0; expert < n_expert; ++expert) {
            double tokens_for_expert = E_ptr[src_device * n_expert + expert];
            if (tokens_for_expert == 0) continue;
            
            if (expert_locations[expert].empty()) continue;
        
            /* 避免重复分配临时向量，直接计算 */
            int intra_count = 0, inter_count = 0;
            const auto& locations = expert_locations[expert];
            const size_t num_locations = locations.size();
            
            // 第一遍：统计节点内外的总权重
            for (size_t i = 0; i < num_locations; ++i) {
                const auto& loc = locations[i];
                int target_node = loc.device >> 3; // 优化: 位运算替代除法
                if (target_node == src_node) {
                    intra_count += loc.weight;
                } else {
                    inter_count += loc.weight;
                }
            }
        
            double remaining_tokens = tokens_for_expert;
            
            /* 预计算常数因子 */
            const double comp_factor = v_comp / 1000;
            const double token_factor = M_token * 1e-9;
            
            /* 阶段1: 节点内路由 (均匀分布) */
            if (intra_count > 0) {
                const double tokens_per_weight = remaining_tokens / intra_count;
                double extra_tokens = std::fmod(remaining_tokens, intra_count);
                
                for (size_t i = 0; i < num_locations; ++i) {
                    const auto& loc = locations[i];
                    int target_node = loc.device >> 3;
                    if (target_node == src_node) {
                        double tokens_to_assign = tokens_per_weight * loc.weight;
                        double extra = std::min(static_cast<double>(loc.weight), extra_tokens);
                        tokens_to_assign += extra;
                        extra_tokens -= extra;
                        
                        S_ptr[src_device * n_expert * n_device + expert * n_device + loc.device] += tokens_to_assign;
                        
                        /* 同时计算时间，使用预计算的常数 */
                        double bw = bandwidth_function(src_device, loc.device, v_intra, v_inter);
                        comm_times[src_device] += token_factor * tokens_to_assign / bw;
                        comp_times[loc.device] += tokens_to_assign * comp_factor;
                    }
                }
                remaining_tokens = 0;
            }
            
            /* 阶段2: 节点间路由 (均匀分布剩余token) */
            if (remaining_tokens > 0 && inter_count > 0) {
                const double tokens_per_weight = remaining_tokens / inter_count;
                double extra_tokens = std::fmod(remaining_tokens, inter_count);
                
                for (size_t i = 0; i < num_locations; ++i) {
                    const auto& loc = locations[i];
                    int target_node = loc.device >> 3;
                    if (target_node != src_node) {
                        double tokens_to_assign = tokens_per_weight * loc.weight;
                        double extra = std::min(static_cast<double>(loc.weight), extra_tokens);
                        tokens_to_assign += extra;
                        extra_tokens -= extra;
                        
                        S_ptr[src_device * n_expert * n_device + expert * n_device + loc.device] += tokens_to_assign;
                        
                        /* 同时计算时间，使用预计算的常数 */
                        double bw = bandwidth_function(src_device, loc.device, v_intra, v_inter);
                        comm_times[src_device] += token_factor * tokens_to_assign / bw;
                        comp_times[loc.device] += tokens_to_assign * comp_factor;
                    }
                }
            }
        }
    }
    
    /* 计算总时间 */
    double total_comm = std::accumulate(comm_times.begin(), comm_times.end(), 0.0);
    double max_comp = *std::max_element(comp_times.begin(), comp_times.end());
    double total_time = 4 * total_comm + 3 * max_comp;
    
    return std::make_pair(S_np, total_time);
}

/* ---------- 5a. _calculate_total_time (保持向后兼容) ---------- */
double calculate_total_time(int n_device, int n_expert, 
                           const py::array_t<double>& S_np, double M_token,
                           double v_comp = 0.001484375,
                           double v_intra = 216.269,
                           double v_inter = 88.9539) {
    py::buffer_info S_buf = S_np.request();
    if (S_buf.ndim != 3 || S_buf.shape[0] != n_device || 
        S_buf.shape[1] != n_expert || S_buf.shape[2] != n_device)
        throw std::runtime_error("S shape mismatch");
    
    const double* S_ptr = static_cast<double*>(S_buf.ptr);
    
    std::vector<double> comm_times(n_device, 0.0);
    std::vector<double> comp_times(n_device, 0.0);
    
    for (int i = 0; i < n_device; ++i) {
        /* 通信时间 */
        for (int j = 0; j < n_expert; ++j) {
            for (int k = 0; k < n_device; ++k) {
                double tokens = S_ptr[i * n_expert * n_device + j * n_device + k];
                if (tokens > 0) {
                    double bw = bandwidth_function(i, k, v_intra, v_inter);
                    comm_times[i] += M_token * tokens / bw * 1e-9;
                }
            }
        }
        
        /* 计算时间 */
        for (int k = 0; k < n_device; ++k) {
            for (int j = 0; j < n_expert; ++j) {
                double tokens = S_ptr[k * n_expert * n_device + j * n_device + i];
                if (tokens > 0) {
                    comp_times[i] += tokens * v_comp / 1000;
                }
            }
        }
    }
    
    double total_comm = std::accumulate(comm_times.begin(), comm_times.end(), 0.0);
    double max_comp = *std::max_element(comp_times.begin(), comp_times.end());
    
    return 4 * total_comm + 3 * max_comp;
}

/* ---------- 6. greedy_load_balancing_heuristic ---------- */
py::tuple greedy_load_balancing_heuristic(int n_device,
                                          int n_expert,
                                          const py::array_t<double>& E_np,
                                          int C_e) {
    /* 解析 numpy 二维数组 E */
    py::buffer_info buf = E_np.request();
    if (buf.ndim != 2 || buf.shape[0] != n_device || buf.shape[1] != n_expert)
        throw std::runtime_error("E shape mismatch");

    const double* E_ptr = static_cast<double*>(buf.ptr);
    std::vector<double> expert_loads(n_expert, 0.0);
    for (int j = 0; j < n_expert; ++j)
        for (int i = 0; i < n_device; ++i)
            expert_loads[j] += E_ptr[i * n_expert + j];

    /* 保存最优方案 */
    GreedyResult best;

    /* 1) 平均副本策略 */
    if ((C_e * n_device) % n_expert == 0) {
        int r = C_e * n_device / n_expert;
        std::vector<int> replicas(n_expert, r);
        best = get_greedy_placement(replicas, expert_loads, n_device, n_expert, C_e);
    }

    /* 2) 精确副本策略 */
    auto replicas = allocate_expert_replicas_precise(expert_loads, n_device, C_e);
    auto res = get_greedy_placement(replicas, expert_loads, n_device, n_expert, C_e);
    if (best.A.empty() || res.max_load < best.max_load)
        best = std::move(res);

    /* 3) 生成 Python list[list[int]] A_res */
    py::list A_res;
    for (int j = 0; j < n_device; ++j) {
        py::list tmp;
        for (int i = 0; i < n_expert; ++i) {
            for (int cnt = 0; cnt < best.A[i][j]; ++cnt)
                tmp.append(i);
        }
        A_res.append(tmp);
    }

    /* 返回 (0, 0, A_res) 以兼容原接口 */
    return py::make_tuple(0, 0, A_res);
}

/* ---------- 7. 新的完整接口 ---------- */
py::tuple greedy_load_balancing_heuristic_complete(int n_device,
                                                   int n_expert,
                                                   const py::array_t<double>& E_np,
                                                   int C_e,
                                                   double M_token,
                                                   int num_perturbations = 0,
                                                   double v_comp = 0.001484375,
                                                   double v_intra = 216.269,
                                                   double v_inter = 88.9539) {
    /* 解析 numpy 二维数组 E */
    py::buffer_info buf = E_np.request();
    if (buf.ndim != 2 || buf.shape[0] != n_device || buf.shape[1] != n_expert)
        throw std::runtime_error("E shape mismatch");

    const double* E_ptr = static_cast<double*>(buf.ptr);
    std::vector<double> expert_loads(n_expert, 0.0);
    for (int j = 0; j < n_expert; ++j)
        for (int i = 0; i < n_device; ++i)
            expert_loads[j] += E_ptr[i * n_expert + j];

    /* 保存最优方案 */
    GreedyResult best;
    double best_total_time = std::numeric_limits<double>::max();
    
    /* 收集所有要测试的replicas方案 */
    std::vector<std::vector<int>> all_replicas_candidates;
    
    /* 1) 平均副本策略 */
    if ((C_e * n_device) % n_expert == 0) {
        int r = C_e * n_device / n_expert;
        std::vector<int> avg_replicas(n_expert, r);
        all_replicas_candidates.push_back(avg_replicas);
    }

    /* 2) 精确副本策略 */
    auto precise_replicas = allocate_expert_replicas_precise(expert_loads, n_device, C_e);
    all_replicas_candidates.push_back(precise_replicas);
    
    // 对精确副本策略进行扰动
    auto perturbed_precise = generate_perturbed_replicas(precise_replicas, n_device, num_perturbations);
    for (size_t i = 1; i < perturbed_precise.size(); ++i) { // 跳过第一个（原始方案）
        all_replicas_candidates.push_back(perturbed_precise[i]);
    }
    
    // 如果有平均副本策略，也对其进行扰动
    if ((C_e * n_device) % n_expert == 0) {
        int r = C_e * n_device / n_expert;
        std::vector<int> avg_replicas(n_expert, r);
        auto perturbed_avg = generate_perturbed_replicas(avg_replicas, n_device, num_perturbations);
        for (size_t i = 1; i < perturbed_avg.size(); ++i) { // 跳过第一个（原始方案）
            all_replicas_candidates.push_back(perturbed_avg[i]);
        }
    }
    
    /* 4) 枚举所有方案并找到最优解 */
    for (const auto& replicas : all_replicas_candidates) {
        try {
            auto result = get_greedy_placement(replicas, expert_loads, n_device, n_expert, C_e);
            
            /* 生成numpy数组A */
            std::vector<size_t> A_shape = {static_cast<size_t>(n_expert), static_cast<size_t>(n_device)};
            auto A_np = py::array_t<double>(A_shape);
            py::buffer_info A_buf = A_np.request();
            double* A_ptr = static_cast<double*>(A_buf.ptr);
            
            for (int i = 0; i < n_expert; ++i) {
                for (int j = 0; j < n_device; ++j) {
                    A_ptr[i * n_device + j] = static_cast<double>(result.A[i][j]);
                }
            }

            /* 生成路由矩阵并计算总时间 (融合操作) */
            auto [S_np, total_time] = generate_smart_routing_and_calculate_time(
                n_device, n_expert, E_np, A_np, M_token, v_comp, v_intra, v_inter);
            
            if (total_time < best_total_time) {
                best = std::move(result);
                best_total_time = total_time;
            }
        } catch (const std::exception& e) {
            // 某些replicas方案可能不可行，跳过即可
            continue;
        }
    }
    
    /* 检查是否找到了有效解 */
    if (best.A.empty()) {
        throw std::runtime_error("No valid placement found");
    }

    /* 5) 生成 Python list[list[int]] A_res */
    py::list A_res;
    for (int j = 0; j < n_device; ++j) {
        py::list tmp;
        for (int i = 0; i < n_expert; ++i) {
            for (int cnt = 0; cnt < best.A[i][j]; ++cnt)
                tmp.append(i);
        }
        A_res.append(tmp);
    }

    /* 返回 (0, 0, A_res) 以兼容原接口 */
    return py::make_tuple(0, 0, A_res);
}

/* ---------- Python 绑定 ---------- */
PYBIND11_MODULE(greedy_balancer, m) {
    m.doc() = "Greedy load balancing heuristic (full C++)";
    m.def("greedy_load_balancing_heuristic", &greedy_load_balancing_heuristic,
          py::arg("n_device"),
          py::arg("n_expert"),
          py::arg("E"),
          py::arg("C_e"));
    m.def("greedy_load_balancing_heuristic_complete", &greedy_load_balancing_heuristic_complete,
          py::arg("n_device"),
          py::arg("n_expert"),
          py::arg("E"),
          py::arg("C_e"),
          py::arg("M_token"),
          py::arg("num_perturbations") = 0,
          py::arg("v_comp") = 0.001484375,
          py::arg("v_intra") = 216.269,
          py::arg("v_inter") = 88.9539);
    m.def("generate_smart_routing", &generate_smart_routing,
          py::arg("n_device"),
          py::arg("n_expert"),
          py::arg("E"),
          py::arg("A"),
          py::arg("v_intra") = 216.269,
          py::arg("v_inter") = 88.9539);
    m.def("generate_smart_routing_and_calculate_time", &generate_smart_routing_and_calculate_time,
          py::arg("n_device"),
          py::arg("n_expert"),
          py::arg("E"),
          py::arg("A"),
          py::arg("M_token"),
          py::arg("v_comp") = 0.001484375,
          py::arg("v_intra") = 216.269,
          py::arg("v_inter") = 88.9539);
    m.def("calculate_total_time", &calculate_total_time,
          py::arg("n_device"),
          py::arg("n_expert"),
          py::arg("S"),
          py::arg("M_token"),
          py::arg("v_comp") = 0.001484375,
          py::arg("v_intra") = 216.269,
          py::arg("v_inter") = 88.9539);
}