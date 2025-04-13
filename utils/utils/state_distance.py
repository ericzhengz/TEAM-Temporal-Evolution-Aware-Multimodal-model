import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class AdaptiveStateDistanceMatrix(nn.Module):
    """自适应虫态距离矩阵
    
    在训练过程中动态学习不同虫态之间的相对距离关系，
    用于优化UniCL对比损失中的正负样本构造。
    """
    def __init__(self, num_states=10, feature_dim=512, init_with_prior=True, 
                 update_interval=10, decay_factor=0.9):
        super().__init__()
        self.num_states = num_states
        self.feature_dim = feature_dim
        
        # 初始化距离矩阵
        if init_with_prior:
            # 使用先验知识初始化 - 使用预定义的虫态关系
            init_matrix = torch.ones(num_states, num_states)
            # 设置一些典型虫态间的距离
            # 1=larva, 0=egg, 3=nymph, 4=adult, 2=pupa
            
            # 成虫(4)和幼虫(1)距离最远
            init_matrix[1, 4] = init_matrix[4, 1] = 2.0
            
            # 若虫(3)和成虫(4)距离较近
            init_matrix[3, 4] = init_matrix[4, 3] = 0.7
            
            # 幼虫(1)和蛹(5)距离中等
            init_matrix[1, 2] = init_matrix[2, 1] = 1.5
            
            # 卵(2)和所有其他状态距离较远
            init_matrix[0, :] = init_matrix[:, 0] = 1.8
            init_matrix[0, 0] = 1.0  # 对角线设为1
        else:
            # 无先验知识，初始为单位矩阵(对角线为1，其他位置也为1)
            init_matrix = torch.ones(num_states, num_states)
            
        # 可学习的距离矩阵参数 
        self.distance_factors = nn.Parameter(init_matrix)
        
        # 状态嵌入投影，用于预测虫态间关系
        self.state_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
        )
        
        # 损失历史记录
        self.update_history = []
        
        # 添加更新控制参数
        self.update_interval = update_interval  # 每隔多少批次更新一次
        self.decay_factor = decay_factor        # 移动平均因子
        self.update_counter = 0                 # 批次计数器
        self.is_training = True                 # 训练模式标志
        
    def get_state_distance(self, state_i, state_j):
        """获取两个虫态之间的距离"""
        return self.distance_factors[state_i, state_j]
    
    def get_distance_matrix(self):
        """获取当前的距离矩阵"""
        # 确保对称性
        sym_matrix = (self.distance_factors + self.distance_factors.t()) / 2
        # 确保对角线为1
        eye_mask = torch.eye(self.num_states, device=self.distance_factors.device)
        return sym_matrix * (1 - eye_mask) + eye_mask
    
    def train(self, mode=True):
        """重写train方法，控制是否更新距离矩阵"""
        super().train(mode)
        self.is_training = mode
        return self
        
    def forward(self, state_features, state_ids):
        """
        根据当前批次的状态特征动态更新距离矩阵
        state_features: [batch_size, feature_dim] - 虫态特征向量
        state_ids: [batch_size] - 虫态ID
        """
        # 获取当前距离矩阵
        current_matrix = self.get_distance_matrix()
        
        # 只有在训练模式且达到更新间隔时才更新矩阵
        if self.is_training and self.update_counter % self.update_interval == 0:
            batch_size = state_features.size(0)
            device = state_features.device
            
            # 将特征投影到低维空间
            projected_features = self.state_projector(state_features)
            
            # 计算每个虫态的特征中心
            state_centers = {}
            for state_id in range(1, self.num_states):
                mask = (state_ids == state_id)
                if mask.sum() > 0:  # 如果该虫态在当前批次中存在
                    center = state_features[mask].mean(0)
                    state_centers[state_id] = center
            
            # 记录更新信息
            update_info = {}
            
            # 根据特征相似性动态更新距离矩阵
            if len(state_centers) > 1:
                # 创建虫态特征矩阵
                state_ids_list = sorted(state_centers.keys())
                centers_tensor = torch.stack([state_centers[i] for i in state_ids_list])
                
                # 计算成对余弦相似度
                sim_matrix = torch.mm(F.normalize(centers_tensor, dim=1), 
                                      F.normalize(centers_tensor, dim=1).t())
                
                # 转换相似度到距离 (相似度高→距离小)
                distance_matrix = 2.0 - sim_matrix  # 范围[1,2]
                
                # 更新距离矩阵
                for i, s_i in enumerate(state_ids_list):
                    for j, s_j in enumerate(state_ids_list):
                        if i != j:
                            # 使用更大的decay_factor降低更新幅度
                            old_dist = self.distance_factors[s_i, s_j].item()
                            new_dist = self.decay_factor * old_dist + (1 - self.decay_factor) * distance_matrix[i, j].item()
                            
                            # 记录重要的更新
                            if abs(new_dist - old_dist) > 0.1:
                                update_info[(s_i, s_j)] = (old_dist, new_dist)
                            
                            # 更新距离矩阵
                            self.distance_factors.data[s_i, s_j] = new_dist
                            self.distance_factors.data[s_j, s_i] = new_dist  # 保持对称
        
            # 记录更新历史
            if update_info:
                self.update_history.append(update_info)
        
        # 更新计数器
        self.update_counter += 1
        
        # 返回当前的距离矩阵(无论是否更新)
        return current_matrix
    
    def visualize_distance_matrix(self, save_path=None):
        """可视化虫态距离矩阵"""
        plt.figure(figsize=(8, 6))
        
        # 获取当前距离矩阵 - 复制到CPU
        distance_mat = self.get_distance_matrix().detach().cpu().numpy()
        
        # 绘制热图
        im = plt.imshow(distance_mat, cmap='viridis')
        plt.colorbar(im)
        
        # 添加标签
        state_names = {
            1: "larva", 0: "egg", 3: "nymph", 
            4: "adult", 5: "other", 2: "pupa"
        }
        
        # 设置坐标轴
        ticks = np.arange(self.num_states)
        plt.xticks(ticks, [state_names.get(i, str(i)) for i in range(self.num_states)])
        plt.yticks(ticks, [state_names.get(i, str(i)) for i in range(self.num_states)])
        
        # 添加文本标注
        for i in range(self.num_states):
            for j in range(self.num_states):
                if i != j:  # 只显示非对角元素的值
                    plt.text(j, i, f"{distance_mat[i, j]:.2f}", 
                            ha="center", va="center", 
                            color="white" if distance_mat[i, j] > 1.5 else "black")
        
        plt.title("虫态距离矩阵")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def integrate_with_state_distance(self, state_distance_matrix):
        """整合虫态距离矩阵信息到演化图网络中"""
        # 从距离矩阵中提取关系强度信息
        try:
            distance = state_distance_matrix.get_distance_matrix().detach()
            similarity = 1.0 - torch.clamp(distance / distance.max(), min=0.0, max=1.0)
            
            # 调整演化投影器的偏置
            for key in self.evolution_projector:
                src_state, dst_state = map(int, key.split('-'))
                if src_state < similarity.shape[0] and dst_state < similarity.shape[0]:
                    sim_value = similarity[src_state, dst_state].item()
                    
                    # 对投影器最后一层的偏置做调整
                    last_layer = self.evolution_projector[key][-1]
                    if hasattr(last_layer, 'bias') and last_layer.bias is not None:
                        with torch.no_grad():
                            scale = 0.1 * sim_value
                            last_layer.bias.data *= (1.0 - scale)
            return True
        except Exception as e:
            print(f"整合虫态距离矩阵失败: {e}")
            return False