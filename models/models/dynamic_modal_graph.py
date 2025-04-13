import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicRelationModeler(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_relations=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        
        # 关系投影器，用于衡量不同特征间的关系强度
        self.relation_projector = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim * 2, hidden_dim), 
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_relations)
        ])
        
        # 拓扑控制门，决定是否建立边连接
        self.topology_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 全局特征聚合器(新增)
        self.global_aggregator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, features_dict, task_id=0, relation_threshold=0.2):
        """构建多模态动态关系图
        
        参数:
            features_dict: 不同模态特征字典 {modal_type: tensor}
            task_id: 当前任务ID
            relation_threshold: 关系阈值，高于此值才建立连接
            
        返回:
            node_features: 节点特征 [num_nodes, feature_dim]
            edge_index: 边连接 [2, num_edges]
            edge_weights: 边权重 [num_edges]
            node_types: 节点类型列表
        """
        # 收集所有节点特征和类型
        node_features = []
        node_types = []
        for feat_type, features in features_dict.items():
            node_features.append(features)
            node_types.extend([feat_type] * features.shape[0])
        
        node_features = torch.cat(node_features, dim=0)
        device = node_features.device
        
        # 构建边连接和权重
        edge_index = []
        edge_weights = []
        edge_types = []  # 新增：记录每条边的类型
        
        # 构建完全图并评估每对节点间的关系
        for i in range(len(node_features)):
            for j in range(len(node_features)):
                if i == j:
                    continue
                
                # 拼接特征对计算关系得分
                pair_feat = torch.cat([node_features[i], node_features[j]], dim=0)
                # 计算各种关系得分
                relation_scores = torch.cat([proj(pair_feat) for proj in self.relation_projector])
                
                # 找出最强关系及其分数
                max_rel_idx = torch.argmax(relation_scores)
                best_rel_score = relation_scores[max_rel_idx]
                
                # 如果关系强度超过阈值，则建立边
                if best_rel_score > relation_threshold:
                    edge_index.append([i, j])
                    edge_weights.append(best_rel_score.item())
                    edge_types.append(max_rel_idx.item())  # 记录关系类型
        
        if not edge_index:  # 处理没有边的情况
            return node_features, None, None, node_types, None
        
        # 转换为PyTorch张量
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float, device=device)
        edge_types = torch.tensor(edge_types, dtype=torch.long, device=device)
        
        return node_features, edge_index, edge_weights, node_types, edge_types

    def get_global_feature(self, node_features):
        """提取全局图特征"""
        if node_features is None or len(node_features) == 0:
            return None
        
        # 简单聚合所有节点特征
        pooled = torch.mean(node_features, dim=0, keepdim=True)
        return self.global_aggregator(pooled)

class DynamicGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # 输入层
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
        # 输出层
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.norms.append(nn.LayerNorm(out_dim))
        self.dropouts.append(nn.Dropout(dropout))
    
    def forward(self, x, edge_index=None, edge_weights=None):
        """
        执行图卷积
        
        如果提供了边信息，则执行消息传递；否则只执行节点特征变换
        """
        if edge_index is None:  # 无边情况
            for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts):
                x = dropout(norm(F.relu(layer(x))))
            return x
            
        # 有边情况：执行消息传递
        for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts):
            # 节点特征变换
            h = layer(x)
            h = F.relu(h)
            
            # 消息传递(简化版图卷积) —— 避免 in-place 操作
            if edge_index is not None and edge_weights is not None:
                src, dst = edge_index
                h_updated = h.clone()  # 新建一个副本用于累加消息
                for i in range(src.size(0)):
                    src_idx = src[i]
                    dst_idx = dst[i]
                    weight = edge_weights[i]
                    h_updated[dst_idx] = h_updated[dst_idx] + weight * h[src_idx]
                h = h_updated
                     
            # 规范化和dropout
            h = norm(h)
            h = dropout(h)
            x = h
        return x

class GlobalConnectivityExtractor(nn.Module):
    def __init__(self, feature_dim, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 使用Transformer编码器处理全局关系
        self.connectivity_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=feature_dim * 4,
                dropout=dropout,
                batch_first=True  # 更新参数，确保批次优先
            ),
            num_layers=2
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    
    def forward(self, node_features):
        """提取全局连接特征
        
        参数:
            node_features: 节点特征 [num_nodes, feature_dim]
            
        返回:
            global_features: 全局特征 [1, feature_dim]
        """
        if node_features is None or len(node_features) == 0:
            return None
        
        # 添加批次维度
        batch_features = node_features.unsqueeze(0)  # [1, num_nodes, feature_dim]
        
        # 通过Transformer获取特征间关系
        global_features = self.connectivity_transformer(batch_features)  # [1, num_nodes, feature_dim]
        
        # 全局池化并投影
        pooled = torch.mean(global_features, dim=1)  # [1, feature_dim]
        return self.output_proj(pooled)

class TemporalStateGCN(nn.Module):
    """时序图卷积网络 - 用于建模昆虫完整生命周期"""
    def __init__(self, feature_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 节点特征编码
        self.node_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 时间编码
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )
        
        # 叠加多个时序图卷积块
        self.temporal_blocks = nn.ModuleList([
            TemporalGCNBlock(hidden_dim + hidden_dim // 4) for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim + hidden_dim // 4, feature_dim)
    
    def forward(self, node_features, edge_index, edge_weights, time_steps):
        """
        时序图卷积网络前向传播
        
        参数:
            node_features: 节点特征 [num_nodes, feature_dim]
            edge_index: 边索引 [2, num_edges]
            edge_weights: 边权重 [num_edges]
            time_steps: 每个节点的时间步 [num_nodes, 1]
        """
        # 编码节点特征
        h = self.node_encoder(node_features)
        
        # 编码时间特征
        t = self.time_encoder(time_steps)
        
        # 合并节点特征和时间特征
        h_t = torch.cat([h, t], dim=-1)
        
        # 通过时序图卷积块
        for block in self.temporal_blocks:
            h_t = block(h_t, edge_index, edge_weights)
        
        # 输出投影并归一化
        out_features = self.output_proj(h_t)
        out_features = F.normalize(out_features, dim=-1)
        
        return out_features

class TemporalGCNBlock(nn.Module):
    """时序图卷积基本块"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 消息传递网络
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 节点更新网络
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 时序门控机制
        self.temporal_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_weights):
        """
        时序图卷积块前向传播
        
        参数:
            x: 节点特征 [num_nodes, hidden_dim]
            edge_index: 边索引 [2, num_edges]
            edge_weights: 边权重 [num_edges]
        """
        src, dst = edge_index
        num_nodes = x.size(0)
        
        # 计算消息
        messages = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
        message_counts = torch.zeros(num_nodes, 1, device=x.device)
        
        # 创建边的消息
        for i in range(src.size(0)):
            s, d = src[i], dst[i]
            w = edge_weights[i]
            
            # 构建消息（源节点特征与目标节点特征拼接）
            edge_message = torch.cat([x[s], x[d]], dim=-1)
            edge_message = self.message_net(edge_message) * w
            
            # 累加消息到目标节点
            messages[d] += edge_message
            message_counts[d] += 1
        
        # 平均聚合消息(避免节点度数不同导致的影响)
        valid_mask = (message_counts > 0).float()
        messages = messages / (message_counts + 1e-8) * valid_mask
        
        # 计算时序门控权重
        temporal_weights = self.temporal_gate(x)
        
        # 更新节点特征
        combined = torch.cat([x, messages], dim=-1)
        h_new = self.update_net(combined)
        
        # 应用门控机制
        h_updated = temporal_weights * h_new + (1 - temporal_weights) * x
        
        return h_updated
