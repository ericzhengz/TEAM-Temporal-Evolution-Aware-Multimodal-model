import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dynamic_modal_graph import DynamicGCN, GlobalConnectivityExtractor
import os

class InsectLifecycleModel(nn.Module):
    """昆虫生命周期建模器 - 虫态特征嵌入与演化"""
    def __init__(self, feature_dim, hidden_dim, num_states=10):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_states = num_states
        
        # 虫态嵌入
        self.state_embeddings = nn.Embedding(num_states, feature_dim)
        
        # 虫态类型映射表
        self.state_type_names = {
            0: "egg", 1: "larva", 2: "pupa", 3: "nymph", 4: "adult", 5: "other"
        }
        
        # 昆虫类别的生命周期类型
        self.class_lifecycle_types = {}
        
        # 时序图卷积网络 - 新增
        from models.dynamic_modal_graph import TemporalStateGCN
        self.temporal_gcn = TemporalStateGCN(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )
        
        # 演化投影器字典
        self.evolution_projector = nn.ModuleDict()
        
        # 虫态演化类型检测器
        self.evolution_detector = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3种生命周期类型：完全变态、不完全变态、直接发育
            nn.Softmax(dim=1)
        )
        
    def get_state_embeddings(self, state_ids):
        """替代原来的虫态嵌入查询功能"""
        return self.state_embeddings(state_ids)
        
    def get_distance(self, state_id1, state_id2):
        """替代原来的虫态距离查询"""
        return self.state_distance_matrix[state_id1, state_id2]
        
    def _detect_evolution_type(self, state_ids):
        """检测当前昆虫的演化路径类型"""
        if 1 in state_ids and 4 in state_ids:  # 同时有幼虫和成虫
            return 'larvae_to_adult'
        elif 3 in state_ids and 4 in state_ids:  # 同时有若虫和成虫
            return 'nymph_to_adult'
        elif 1 in state_ids:  # 只有幼虫
            return 'larvae_to_adult'
        elif 3 in state_ids:  # 只有若虫
            return 'nymph_to_adult'
        elif 4 in state_ids:  # 只有成虫
            return 'adult_only'
        else:
            return 'unknown'
            
    def _build_evolution_graph(self, states, protos):
        """构建虫态演化有向图
        
        参数:
            states: 虫态ID列表
            protos: 原型特征字典 {state_id: feature_tensor}
            
        返回:
            node_features: 节点特征 [num_nodes, feature_dim]
            edge_index: 边连接 [2, num_edges]
            edge_weights: 边权重 [num_edges]
        """
        if not states or len(protos) < 2:
            return None, None, None
            
        device = next(iter(protos.values())).device
        
        # 检测演化类型
        evo_type = self._detect_evolution_type(states)
        if evo_type == 'unknown' or evo_type == 'adult_only':
            return None, None, None
            
        # 构建节点特征和边
        node_features = []
        edge_src, edge_dst = [], []
        edge_weights = []
        state_to_idx = {}
        
        # 添加节点
        for idx, state_id in enumerate(states):
            if state_id in protos:
                node_features.append(protos[state_id])
                state_to_idx[state_id] = idx
                
        # 添加有向边
        if evo_type == 'larvae_to_adult' and 1 in state_to_idx and 4 in state_to_idx:
            src_idx = state_to_idx[1]
            dst_idx = state_to_idx[4]
            edge_src.append(src_idx)
            edge_dst.append(dst_idx)
            edge_weights.append(1.0)
            
        elif evo_type == 'nymph_to_adult' and 3 in state_to_idx and 4 in state_to_idx:
            src_idx = state_to_idx[3]
            dst_idx = state_to_idx[4]
            edge_src.append(src_idx)
            edge_dst.append(dst_idx)
            edge_weights.append(1.0)
        
        # 转换为PyTorch张量
        if not node_features or not edge_src:
            return None, None, None
            
        node_features = torch.stack(node_features)
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(device)
        edge_weights = torch.tensor(edge_weights).to(device)
        
        return node_features, edge_index, edge_weights
    
    def model_evolution_trajectory(self, src_state_id, src_feat, dst_state_id=4):
        """模拟从源虫态到目标虫态的特征演化轨迹
        
        参数:
            src_state_id: 源虫态ID (1表示幼虫, 3表示若虫)
            src_feat: 源虫态特征 [feature_dim]
            dst_state_id: 目标虫态ID (默认为4，成虫)
            
        返回:
            trajectory_feats: 演化轨迹特征 [num_steps, feature_dim]
            attention_weights: 注意力权重 [num_steps]
        """
        if dst_state_id != 4 or (src_state_id != 1 and src_state_id != 3):
            return None, None
            
        key = f"{src_state_id}-{dst_state_id}"
        if key not in self.evolution_projector:
            return None, None
            
        # 获取演化投影器
        projector = self.evolution_projector[key]
        
        # 生成10步演化轨迹
        num_steps = 10
        trajectory_feats = []
        attention_weights = []
        
        current_feat = src_feat
        for i in range(num_steps):
            # 计算演化步长
            if i == 0:
                trajectory_feats.append(current_feat)
                attention_weights.append(0.0)
                continue
                
            # 逐步演化
            alpha = i / (num_steps - 1)
            delta = projector(current_feat) * (alpha / 2)
            evolved_feat = current_feat + delta
            evolved_feat = F.normalize(evolved_feat, dim=0)
            
            trajectory_feats.append(evolved_feat)
            attention_weights.append(alpha)
            
            # 更新当前特征
            if i < num_steps - 1:
                current_feat = evolved_feat
        
        return torch.stack(trajectory_feats), torch.tensor(attention_weights, device=src_feat.device)
    
    def forward(self, class_prototypes_by_state):
        """
        对每个类别的虫态原型进行演化建模
        
        参数:
            class_prototypes_by_state: {class_id: {state_id: prototype}}
            
        返回:
            evolved_prototypes: 演化后的原型
            evolution_features: 类别级演化特征 [num_classes, feature_dim]
        """
        evolved_prototypes = {}
        evolution_features = []
        
        # 处理每个类别
        for class_id, state_protos in class_prototypes_by_state.items():
            if not state_protos or len(state_protos) < 2:
                evolved_prototypes[class_id] = {k: v.clone() for k, v in state_protos.items()}
                # 对于单一虫态，仍需返回一个代表特征
                if state_protos:
                    evolution_features.append(next(iter(state_protos.values())))
                continue
                
            # 获取该类别的所有虫态
            states = list(state_protos.keys())
            
            # 构建该类别的虫态演化图
            node_feats, edge_idx, edge_weights = self._build_evolution_graph(states, state_protos)
            
            if node_feats is None:
                # 无法构建有效图，保持原型不变
                evolved_prototypes[class_id] = {k: v.clone() for k, v in state_protos.items()}
                if state_protos:
                    evolution_features.append(next(iter(state_protos.values())))
                continue
                
            # 通过图卷积网络进行信息传递
            updated_features = self.gcn(node_feats, edge_idx, edge_weights)
            
            # 提取全局演化特征
            global_feature = self.global_extractor(updated_features)
            evolution_features.append(global_feature.squeeze(0))
            
            # 更新原型
            result_protos = {}
            for i, state_id in enumerate(states):
                if state_id in state_protos:
                    result_protos[state_id] = F.normalize(updated_features[i], dim=0)
                    
            evolved_prototypes[class_id] = result_protos
            
        # 合并所有类别的演化特征
        if evolution_features:
            evolution_features = torch.stack(evolution_features)
        else:
            first_class = next(iter(class_prototypes_by_state.values()), {})
            first_proto = next(iter(first_class.values()), None)
            device = first_proto.device if first_proto is not None else torch.device('cpu')
            evolution_features = torch.zeros(0, self.feature_dim, device=device)
            
        return evolved_prototypes, evolution_features
        
    def evolve_and_update(self, class_prototypes_by_state, epoch=None, max_epoch=None):
        """利用时序图卷积网络更新虫态原型和演化特征"""
        result = {
            'prototypes': class_prototypes_by_state.copy(),
            'embeddings': [],
            'lifecycle_features': {},
            'distances': {}
        }
        
        # 如果类别数量不足，直接返回
        if len(class_prototypes_by_state) < 1:
            return result
            
        # 收集所有类别的虫态特征
        class_state_features = {}
        all_nodes = []
        all_node_classes = []
        all_node_states = []
        all_time_steps = []
        
        # 为每个类构建时序图
        for class_id, state_dict in class_prototypes_by_state.items():
            if len(state_dict) < 2:
                continue
                
            # 检测演化类型
            state_ids = sorted(list(state_dict.keys()))
            lifecycle_type = self._detect_evolution_type(state_ids)
            self.class_lifecycle_types[class_id] = lifecycle_type
            
            # 规范化时间步: 从0到1
            state_to_time = {}
            for idx, state_id in enumerate(state_ids):
                state_to_time[state_id] = idx / max(1, len(state_ids) - 1)
            
            # 收集节点特征
            for state_id, proto in state_dict.items():
                all_nodes.append(proto)
                all_node_classes.append(class_id)
                all_node_states.append(state_id)
                all_time_steps.append([state_to_time[state_id]])
                
            # 当前类的生命周期特征
            lifecycle_features = torch.cat([state_dict[s].unsqueeze(0) for s in state_ids], dim=0).mean(0)
            result['lifecycle_features'][class_id] = lifecycle_features
            
        if not all_nodes:
            return result
            
        # 将所有节点堆叠成一个批次
        node_features = torch.stack(all_nodes)
        node_classes = torch.tensor(all_node_classes, device=node_features.device)
        node_states = torch.tensor(all_node_states, device=node_features.device)
        time_steps = torch.tensor(all_time_steps, device=node_features.device)
        
        # 构建类内和类间边
        edge_index = []
        edge_weights = []
        
        # 添加类内的时序边（遵循虫态发展方向）
        for i in range(len(all_nodes)):
            for j in range(len(all_nodes)):
                if i != j and all_node_classes[i] == all_node_classes[j]:
                    # 只在同一类内部建立边
                    if all_time_steps[i][0] < all_time_steps[j][0]:  # 保证时序方向
                        edge_index.append([i, j])
                        # 边权重与时间差成反比
                        weight = 1.0 - abs(all_time_steps[i][0] - all_time_steps[j][0])
                        edge_weights.append(weight)
        
        # 添加类间的相同虫态边（仅在相同虫态之间）
        for i in range(len(all_nodes)):
            for j in range(len(all_nodes)):
                if i != j and all_node_classes[i] != all_node_classes[j] and all_node_states[i] == all_node_states[j]:
                    # 检查是否同类型生命周期
                    if self.class_lifecycle_types.get(all_node_classes[i]) == self.class_lifecycle_types.get(all_node_classes[j]):
                        edge_index.append([i, j])
                        edge_weights.append(0.5)  # 类间边权重较低
        
        if not edge_index:
            return result
            
        # 转换为PyTorch张量
        edge_index = torch.tensor(edge_index, device=node_features.device).t()
        edge_weights = torch.tensor(edge_weights, device=node_features.device)
        
        # 使用时序图卷积网络更新节点特征
        with torch.no_grad():
            updated_features = self.temporal_gcn(node_features, edge_index, edge_weights, time_steps)
        
        # 将更新后的特征分配回各个类别和虫态
        for i, (class_id, state_id) in enumerate(zip(all_node_classes, all_node_states)):
            result['prototypes'][class_id][state_id] = updated_features[i]
        
        # 为每个类提取演化嵌入
        for class_id in result['lifecycle_features'].keys():
            if class_id in result['prototypes']:
                state_dict = result['prototypes'][class_id]
                if len(state_dict) >= 2:
                    # 计算平均演化特征
                    class_embedding = torch.stack(list(state_dict.values())).mean(0)
                    while len(result['embeddings']) <= class_id:
                        result['embeddings'].append(None)
                    result['embeddings'][class_id] = class_embedding
        
        # 更新虫态距离矩阵
        state_distances = {}
        for i, s1 in enumerate(all_node_states):
            if s1 not in state_distances:
                state_distances[s1] = {}
            for j, s2 in enumerate(all_node_states):
                if i != j:
                    # 计算更新后特征的余弦距离
                    sim = F.cosine_similarity(
                        updated_features[i].unsqueeze(0), 
                        updated_features[j].unsqueeze(0)
                    )
                    dist = 1.0 - sim.item()
                    if s2 not in state_distances[s1]:
                        state_distances[s1][s2] = []
                    state_distances[s1][s2].append(dist)
        
        # 平均每对虫态的距离
        for s1 in state_distances:
            for s2 in state_distances[s1]:
                state_distances[s1][s2] = sum(state_distances[s1][s2]) / len(state_distances[s1][s2])
        
        result['distances'] = state_distances
        return result
        
    def _extract_state_distances(self, class_prototypes_by_state):
        """从当前原型中提取虫态距离关系"""
        # 收集所有虫态的原型
        state_to_protos = {}
        for class_id, state_dict in class_prototypes_by_state.items():
            for state_id, proto in state_dict.items():
                if state_id not in state_to_protos:
                    state_to_protos[state_id] = []
                state_to_protos[state_id].append(proto)
        
        # 计算每个虫态的中心原型
        state_centroids = {}
        for state_id, protos in state_to_protos.items():
            if protos:
                centroid = torch.stack(protos).mean(0)
                state_centroids[state_id] = F.normalize(centroid, dim=0)
        
        # 构建距离矩阵
        states = list(state_centroids.keys())
        distance_matrix = torch.zeros((self.state_distance_matrix.shape[0], self.state_distance_matrix.shape[1]),
                                     device=self.state_distance_matrix.device)
        
        # 填充已知虫态间的距离
        for i, s1 in enumerate(states):
            for j, s2 in enumerate(states):
                if i != j and s1 < distance_matrix.shape[0] and s2 < distance_matrix.shape[0]:
                    sim = F.cosine_similarity(state_centroids[s1].unsqueeze(0), state_centroids[s2].unsqueeze(0))
                    distance_matrix[s1, s2] = 1.0 - sim.item()
                    
        return distance_matrix
    
    def visualize_evolution_path(self, class_id, state_protos, save_dir="./results/evolution", use_pca=True):
        """可视化特定类别的虫态演化路径"""
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        import numpy as np
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        
        if len(state_protos) < 2:
            print(f"类别 {class_id} 只有一种虫态，无法可视化演化路径")
            return
            
        # 获取状态ID和对应原型
        states = list(state_protos.keys())
        evo_type = self._detect_evolution_type(states)
        
        # 增强可视化：添加虫态名称映射
        state_name_map = {
            0: "卵",
            1: "幼虫",
            2: "蛹",
            3: "若虫",
            4: "成虫",
            5: "其他"
        }
        
        if evo_type == 'larvae_to_adult' and 1 in states and 4 in states:
            src_id, dst_id = 1, 4  # 幼虫→成虫
            src_name = state_name_map.get(src_id, "幼虫")
            dst_name = state_name_map.get(dst_id, "成虫")
            key = "1-4"
        elif evo_type == 'nymph_to_adult' and 3 in states and 4 in states:
            src_id, dst_id = 3, 4  # 若虫→成虫
            src_name = state_name_map.get(src_id, "若虫")
            dst_name = state_name_map.get(dst_id, "成虫")
            key = "3-4"
        else:
            print(f"类别 {class_id} 的虫态组合不支持可视化")
            return
            
        # 获取原型特征
        src_feat = state_protos[src_id]
        dst_feat = state_protos[dst_id]
        
        # 生成演化轨迹
        trajectory_feats, attention_weights = self.model_evolution_trajectory(src_id, src_feat, dst_id)
        
        # 合并所有特征用于PCA降维
        all_feats = torch.cat([trajectory_feats, dst_feat.unsqueeze(0)], dim=0)
        all_feats_np = all_feats.cpu().detach().numpy()
        
        # PCA降维
        pca = PCA(n_components=2)
        feats_2d = pca.fit_transform(all_feats_np)
        
        # 绘制演化轨迹
        plt.figure(figsize=(10, 8))
        
        # 绘制轨迹点
        cmap = plt.cm.viridis
        sc = plt.scatter(
            feats_2d[:-1, 0], 
            feats_2d[:-1, 1],
            c=np.arange(len(feats_2d)-1),
            cmap=cmap,
            s=100,
            alpha=0.7,
            label=f"{src_name}→{dst_name} 演化轨迹"
        )
        
        # 添加起始点和终点标记
        plt.scatter(
            feats_2d[0, 0], feats_2d[0, 1],
            marker='o', s=200, color='green',
            edgecolors='black', linewidths=2,
            label=src_name
        )
        
        plt.scatter(
            feats_2d[-1, 0], feats_2d[-1, 1],
            marker='*', s=300, color='red',
            edgecolors='black', linewidths=2,
            label=dst_name
        )
        
        # 添加箭头连接相邻点
        for i in range(len(feats_2d)-2):
            plt.arrow(
                feats_2d[i, 0], feats_2d[i, 1],
                feats_2d[i+1, 0] - feats_2d[i, 0],
                feats_2d[i+1, 1] - feats_2d[i, 1],
                head_width=0.02, head_length=0.03,
                fc=cmap(i/len(feats_2d)), ec=cmap(i/len(feats_2d)),
                alpha=0.6
            )
        
        # 最后一个箭头指向目标
        plt.arrow(
            feats_2d[-2, 0], feats_2d[-2, 1],
            feats_2d[-1, 0] - feats_2d[-2, 0],
            feats_2d[-1, 1] - feats_2d[-2, 1],
            head_width=0.03, head_length=0.05,
            fc='red', ec='red',
            alpha=0.9
        )
        
        plt.title(f"类别 {class_id} {src_name}→{dst_name} 演化轨迹", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加颜色条显示演化进度
        cbar = plt.colorbar(sc)
        cbar.set_label('演化进度')
        
        save_path = os.path.join(save_dir, f"class_{class_id}_{key}_evolution.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存虫态演化轨迹: {save_path}")
    
    def integrate_with_state_distance(self, state_distance_matrix):
        """整合虫态距离矩阵信息到演化图网络中
        
        参数:
            state_distance_matrix: AdaptiveStateDistanceMatrix实例
        """
        # 从距离矩阵中提取关系强度信息，用于调整图网络边权重
        distance = state_distance_matrix.get_distance_matrix().detach()
        similarity = 1.0 - torch.clamp(distance / distance.max(), min=0.0, max=1.0)
        
        # 调整演化投影器的偏置
        for key in self.evolution_projector:
            src_state, dst_state = map(int, key.split('-'))
            if src_state < similarity.shape[0] and dst_state < similarity.shape[0]:
                # 从相似度矩阵中提取对应虫态间的相似度
                sim_value = similarity[src_state, dst_state].item()
                
                # 对投影器最后一层的偏置做调整，促进更相似虫态间特征传递
                last_layer = self.evolution_projector[key][-1]
                if hasattr(last_layer, 'bias') and last_layer.bias is not None:
                    with torch.no_grad():
                        scale = 0.1 * sim_value  # 相似度越高，偏置越小(促进特征传递)
                        last_layer.bias.data *= (1.0 - scale)
        
        return True

    # 新增方法：构建类别生命周期图
    def build_class_lifecycle_graph(self, class_id, state_protos):
        """为特定类别构建完整生命周期图
        
        参数:
            class_id: 类别ID
            state_protos: {state_id: prototype} 该类别的所有虫态原型
            
        返回:
            生命周期图数据结构
        """
        if not state_protos or len(state_protos) < 2:
            return None, None, None, None
            
        device = next(iter(state_protos.values())).device
        states = list(state_protos.keys())
        
        # 确定生命周期类型
        cycle_type = self._detect_evolution_type(states)
        self.class_lifecycle_types[class_id] = cycle_type
        
        # 构建节点特征
        node_features = []
        state_ids = []
        for state_id in sorted(states):
            if state_id in state_protos:
                node_features.append(state_protos[state_id])
                state_ids.append(state_id)
        
        if not node_features:
            return None, None, None, None
            
        # 构建有向边 - 基于生命周期排序
        edge_index = []
        edge_weights = []
        
        # 按生命周期阶段排序
        if cycle_type == 'larvae_to_adult':  # 1(幼虫) -> 4(成虫)
            lifecycle_order = sorted([s for s in states if s in [1, 4]])
        elif cycle_type == 'nymph_to_adult':  # 3(若虫) -> 4(成虫)
            lifecycle_order = sorted([s for s in states if s in [3, 4]])
        else:
            lifecycle_order = sorted(states)
            
        # 创建时序有向边
        for i in range(len(lifecycle_order)-1):
            src_idx = state_ids.index(lifecycle_order[i])
            dst_idx = state_ids.index(lifecycle_order[i+1])
            edge_index.append([src_idx, dst_idx])
            # 根据虫态之间的自然时序关系设置权重
            edge_weights.append(1.0)
            
        # 计算时间步 - 归一化到[0,1]区间
        time_steps = []
        for state_id in state_ids:
            if cycle_type == 'larvae_to_adult':
                # 幼虫(1)=0.0, 成虫(4)=1.0
                time_step = 0.0 if state_id == 1 else (1.0 if state_id == 4 else 0.5)
            elif cycle_type == 'nymph_to_adult':
                # 若虫(3)=0.0, 成虫(4)=1.0
                time_step = 0.0 if state_id == 3 else (1.0 if state_id == 4 else 0.5)
            else:
                time_step = state_id / max(states)
            time_steps.append([time_step])
            
        # 转换为Tensor
        node_features = torch.stack(node_features)
        time_steps = torch.tensor(time_steps, dtype=torch.float32, device=device)
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float32, device=device)
        else:
            edge_index = None
            edge_weights = None
            
        return node_features, edge_index, edge_weights, time_steps

    # 新增方法：安全的演化路径可视化函数
    def visualize_evolution_path(self, class_id, prototypes, save_dir=None):
        """安全的演化路径可视化函数"""
        # 确保prototypes列表中所有元素都是有效的Tensor
        if not all(isinstance(p, torch.Tensor) for p in prototypes):
            print(f"警告：类别 {class_id} 的某些原型无效，跳过可视化")
            return
        
        try:
            # 确保保存目录存在
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
            # 创建2D PCA可视化
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            
            # 收集所有原型特征
            features = torch.stack(prototypes).cpu().numpy()
            
            # 应用PCA降维
            pca = PCA(n_components=2)
            reduced_features = pca.fit_transform(features)
            
            # 绘制演化路径
            plt.figure(figsize=(8, 6))
            
            # 绘制点
            for i, xy in enumerate(reduced_features):
                plt.scatter(xy[0], xy[1], s=100, alpha=0.8)
                plt.text(xy[0], xy[1], f"State {i}", fontsize=12)
            
            # 绘制箭头连接
            for i in range(len(reduced_features) - 1):
                plt.arrow(
                    reduced_features[i, 0], reduced_features[i, 1],  # 起点
                    reduced_features[i+1, 0] - reduced_features[i, 0],  # 方向x
                    reduced_features[i+1, 1] - reduced_features[i, 1],  # 方向y
                    head_width=0.05, head_length=0.05, fc='blue', ec='blue', alpha=0.5
                )
            
            plt.title(f"虫态演化路径 - 类别 {class_id}")
            plt.tight_layout()
            
            # 保存或显示图像
            if save_dir:
                plt.savefig(f"{save_dir}/class_{class_id}_evolution.png")
                plt.close()
            else:
                plt.show()
        
        except Exception as e:
            print(f"可视化演化路径时出错: {e}")
            import traceback
            traceback.print_exc()

# 添加兼容层，保持向后兼容
StateEvolutionGraph = InsectLifecycleModel  # 将StateEvolutionGraph作为InsectLifecycleModel的别名