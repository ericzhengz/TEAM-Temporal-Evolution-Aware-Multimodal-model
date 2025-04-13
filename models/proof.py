import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import Proof_Net
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, get_attribute, ClipLoss
from utils.data_manager import LaionData
import math
import matplotlib.pyplot as plt
import os
import copy

num_workers = 8

def unicl_loss(image_features, text_features, state_features, labels, state_ids,
               state_distance=None, temperature=0.07, epoch=None, max_epoch=None, 
               evolution_features=None):
    """三路对比学习损失函数，增强与图神经网络的协同"""
    device = image_features.device
    text_features = text_features.to(device)
    state_features = state_features.to(device)
    labels = labels.to(device)
    state_ids = state_ids.to(device)
    
    # 统一特征维度为 [batch_size, feature_dim]
    if len(text_features.shape) > 2:
        text_features = text_features.view(text_features.shape[0], -1)
    if len(state_features.shape) > 2:
        state_features = state_features.view(state_features.shape[0], -1)
    if len(image_features.shape) > 2:
        image_features = image_features.view(image_features.shape[0], -1)

    # 特征批量大小需一致
    batch_size = image_features.shape[0]
    if batch_size < 2:
        logging.warning(f"批次大小过小({batch_size})，对比学习效果可能受限")
        if batch_size == 1:
            return torch.tensor(0.0, device=image_features.device), {'instance_loss': 0.0, 'category_loss': 0.0}

    # 特征归一化
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    state_features = F.normalize(state_features, dim=1)
    
    # 增强时序演化特征
    if evolution_features is not None and len(evolution_features) > 0:
        # 获取每个样本对应类别的演化特征索引
        evo_indices = labels.cpu().numpy()
        # 创建副本进行增强，避免原位修改
        state_features_enhanced = state_features.clone()
        
        # 按类别组织样本索引
        class_indices = {}
        for i, class_idx in enumerate(evo_indices):
            if class_idx not in class_indices:
                class_indices[class_idx] = []
            class_indices[class_idx].append(i)
            
        # 增强同一类别的虫态特征
        for class_idx, indices in class_indices.items():
            if class_idx < len(evolution_features) and evolution_features[class_idx] is not None:
                evo_feat = evolution_features[class_idx].to(device)
                
                # 同类别内的虫态特征互相加强
                if len(indices) >= 2:
                    # 收集该类所有状态特征
                    class_states = [state_ids[i].item() for i in indices]
                    unique_states = sorted(set(class_states))
                    
                    # 如果有多个虫态，建立时序关系
                    if len(unique_states) >= 2:
                        state_to_time = {s: i/(len(unique_states)-1) for i, s in enumerate(unique_states)}
                        
                        # 按时序组织特征
                        for i, idx in enumerate(indices):
                            state = state_ids[idx].item()
                            time_pos = state_to_time[state]
                            
                            # 根据时序位置混合特征
                            mixture = evo_feat.clone()
                            for j, other_idx in enumerate(indices):
                                if i != j:
                                    other_state = state_ids[other_idx].item()
                                    other_time = state_to_time[other_state]
                                    # 基于时间差的权重
                                    weight = 1.0 - abs(time_pos - other_time)
                                    if weight > 0.3:  # 仅考虑时序相近的虫态
                                        mixture = mixture + weight * 0.2 * state_features[other_idx]
                            
                            # 混合原始特征与演化特征
                            enhanced_feat = 0.7 * state_features[idx] + 0.3 * F.normalize(mixture, dim=0)
                            state_features_enhanced[idx] = F.normalize(enhanced_feat, dim=0)
                else:
                    # 单个样本直接使用演化特征增强
                    for idx in indices:
                        enhanced_feat = 0.8 * state_features[idx] + 0.2 * F.normalize(evo_feat, dim=0)
                        state_features_enhanced[idx] = F.normalize(enhanced_feat, dim=0)
        
        state_features = state_features_enhanced
    
    instance_loss = torch.tensor(0.0, device=device)
    category_loss = torch.tensor(0.0, device=device)

    # 动态温度示例 (可自行调整或删除)
    if epoch is not None and max_epoch is not None:
        progress = float(epoch) / float(max_epoch)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        dynamic_temperature = temperature * (0.5 + 0.5 * cosine_decay)
    else:
        dynamic_temperature = temperature

    # ============ 1. 实例级对比 (image/text/state 同一条样本) ============
    # 将同一 index 的 image / text / state 看作正样本，其余为负样本
    # 简化为 pair-wise 强制正样本靠近，负样本远离

    # 堆叠: [batch_size, 3, D]
    tri_feats = torch.stack([image_features, text_features, state_features], dim=1)

    for i in range(batch_size):
        feats_i = tri_feats[i]  # [3, D]
        # feats_i 与自身其他 2 路映射作为正样本
        # feats_i 与其他样本的 3 路映射作为负样本
        sim_matrix = torch.matmul(feats_i, feats_i.t()) / dynamic_temperature
        # 只算本条数据的 3x3，对角线自己不计算
        # 每条 row 只排除自身

        for row in range(3):
            row_sim = sim_matrix[row]  # [3]
            # positive: 其余 2 路
            # negative: 无 (因为只在自己样本内)
            # 若需要纳入全 batch，可以扩展到 tri_feats.view(batch_size*3, D)，与 feats_i 的相似度

            # 这里示例仅做“排除自身”的 softmax
            mask = torch.ones_like(row_sim, device=device)
            mask[row] = 0
            pos_sum = torch.sum(torch.exp(row_sim * mask))
            all_sum = torch.sum(torch.exp(row_sim))
            if pos_sum > 0 and all_sum > 0:
                instance_loss -= torch.log(pos_sum / (all_sum + 1e-8))

    instance_loss = instance_loss / (3 * batch_size)

    # ============ 2. 类别级对比 (image-image) ============
    labels_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    self_mask = 1 - torch.eye(batch_size, device=labels_matrix.device)
    labels_matrix = labels_matrix * self_mask

    # 如果还考虑同虫态则类似添加 states_matrix
    # 这里仅演示同类相似
    img_img_sim = torch.matmul(image_features, image_features.t()) / dynamic_temperature

    valid_samples = 0
    for i in range(batch_size):
        row_sim = img_img_sim[i]  # [batch_size]
        max_val = torch.max(row_sim)
        exp_sim = torch.exp(row_sim - max_val)
        pos_sim = torch.sum(exp_sim * labels_matrix[i])
        all_sim = torch.sum(exp_sim * self_mask[i])
        if pos_sim > 0 and all_sim > 0:
            category_loss -= torch.log(pos_sim / (all_sim + 1e-8))
            valid_samples += 1

    if valid_samples > 0:
        category_loss = category_loss / valid_samples

    # 自定义多路平衡系数
    instance_weight = 1.0
    category_weight = 0.5
    total_loss = instance_weight * instance_loss + category_weight * category_loss

    # 防 NaN
    if torch.isnan(total_loss):
        logging.error("总损失出现 NaN，尝试用部分损失替代")
        if not torch.isnan(instance_loss):
            total_loss = instance_loss
        elif not torch.isnan(category_loss):
            total_loss = category_loss
        else:
            total_loss = torch.tensor(0.0, device=device)

    return total_loss, {
        'instance_loss': float(instance_loss.item()),
        'category_loss': float(category_loss.item()),
        'temperature': dynamic_temperature
    }

class Learner(BaseLearner):
    """
    对三路投影进行增量学习的核心类，主要修改点是拆分出 forward_for_classification，
    训练时区分对比逻辑与分类逻辑，评估时只用 forward_for_classification(或 _compute_accuracy)。
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._train_transformer = False
        self._network = Proof_Net(args, False)

        self.batch_size = get_attribute(args, "batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.frozen_layers = get_attribute(args, "frozen_layers", None)
        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5)

        self._known_classes = 0
        self.use_cos = get_attribute(args, "use_cos", False)

        # 创建自适应虫态距离矩阵 (若有使用)
        from utils.state_distance import AdaptiveStateDistanceMatrix
        self.state_distance = AdaptiveStateDistanceMatrix(
            num_states=10,
            feature_dim=512,
            init_with_prior=True
        ).to(self._device)

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def cal_prototype(self, trainloader, model):
        """
        计算每个类别（以及虫态）的原型，用于后续原型匹配。
        仅针对图像特征做平均，text/state 原型根据需求自行扩展。
        """
        model = model.eval()
        model = model.to(self._device)
        embedding_list, label_list, state_list = [], [], []

        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                if isinstance(batch[1], dict) and 'stage_id' in batch[1]:
                    _, data_dict, label = batch
                    data = data_dict['image'].to(self._device)
                    states = data_dict['stage_id'].to(self._device)
                else:
                    _, data, label = batch
                    data = data.to(self._device)
                    # 默认成虫
                    states = torch.full((data.size(0),), 4, dtype=torch.long).to(self._device)

                label = label.to(self._device)
                embedding = model.convnet.encode_image(data, normalize=True)

                embedding_list.append(embedding)
                label_list.append(label)
                state_list.append(states)

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        state_list = torch.cat(state_list, dim=0)

        class_list = list(range(self._known_classes, self._total_classes))
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            if len(data_index) > 0:
                embedding = embedding_list[data_index]
                proto = embedding.mean(0)
                self._network.img_prototypes[class_index] = proto.to(self._device)

                # 计算虫态级别原型
                states = state_list[data_index]
                if class_index not in self._network.img_prototypes_by_state:
                    self._network.img_prototypes_by_state[class_index] = {}

                unique_states = torch.unique(states)
                for state_id in unique_states:
                    st_mask = (states == state_id)
                    if st_mask.sum() > 0:
                        state_proto = embedding[st_mask].mean(0)
                        self._network.img_prototypes_by_state[class_index][state_id.item()] = state_proto.to(self._device)

    def incremental_train(self, data_manager):
        # 任务序号 + 1
        self._cur_task += 1
        # 新任务类别范围
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        self._network.update_prototype(self._total_classes)
        self._network.update_context_prompt()
        self._network.extend_task()

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # 多模态数据集
        train_dataset = data_manager.get_multimodal_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="train", appendent=self._get_memory())
        self.train_dataset = train_dataset
        self.data_manager = data_manager

        self._old_network = copy.deepcopy(self._network).to(self._device)
        self._old_network.eval()

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_multimodal_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        # 原型网络训练集
        train_dataset_for_protonet = data_manager.get_multimodal_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.to(self._device)

        # 计算原型
        self.cal_prototype(self.train_loader_for_protonet, self._network)

        # 训练
        self._train_proj_with_replay(self.train_loader, self.test_loader, self.train_loader_for_protonet)

        # 构建回放记忆
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        self.state_distance.update_counter = 0

        # 任务训练结束后，分析虫态演化
        try:
            from utils.analysis import analyze_state_evolution
            analyze_state_evolution(
                self._network, 
                data_manager._class_to_label,
                save_dir=f"./results/task_{self._cur_task}/analysis"
            )
            logging.info(f"已完成虫态演化图谱分析")
        except Exception as e:
            logging.error(f"虫态演化分析失败: {str(e)}")
        
        # 确保最后执行一次演化更新
        self._network.state_evolution_graph.integrate_with_state_distance(self.state_distance)
        final_embeddings = self._network.evolve_state_prototypes()
        logging.info(f"已完成最终虫态演化更新 ({len(final_embeddings) if final_embeddings is not None else 0} 类)")

    def _train_proj_with_replay(self, train_loader, test_loader, train_loader_for_protonet):
        self._train_transformer = True
        self._network.to(self._device)

        # 冻结主干，只训练投影层
        for name, param in self._network.convnet.named_parameters():
            if 'logit_scale' not in name:
                param.requires_grad = False
        self._network.freeze_projection_weight_new()

        if self.args['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = torch.optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epoch, eta_min=self.min_lr)
        cliploss = ClipLoss()

        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]  # 取第一个模板
        total_labels = class_to_label[:self._total_classes]

        # 添加一个预热阶段，先初始化虫态距离矩阵
        print("预热阶段：跳过虫态距离矩阵初始化...")
        # 注释掉或删除原有代码，如果state_distance.update_history是列表而非方法
        # with torch.no_grad():
        #     for i, batch in enumerate(train_loader):
        #         if isinstance(batch[1], dict) and 'stage_id'在 batch[1]:
        #             _, data_dict, targets = batch
        #             state_ids = data_dict['stage_id'].to(self._device)
        #             
        #             # 收集批次中的唯一虫态
        #             unique_states = torch.unique(state_ids).cpu().numpy()
        #             
        #             # 对每对唯一虫态更新距离
        #             for s1_idx in range(len(unique_states)):
        #                 for s2_idx in范围(s1_idx+1, len(unique_states)):
        #                     state_id1 = int(unique_states[s1_idx])
        #                     state_id2 = int(unique_states[s2_idx])
        #                     # 同批次出现的虫态，设定中等距离值
        #                     self.state_distance.update_history(state_id1, state_id2, 0.5, weight=0.1)
        #         
        #         if i >= 5:  # 只需几个批次预热
        #             break

        # 在训练前先整合虫态距离矩阵到演化图网络
        self._network.state_evolution_graph.integrate_with_state_distance(self.state_distance)

        prog_bar = tqdm(range(self.tuned_epoch))
        for epoch in range(self.tuned_epoch):
            self._network.train()
            losses, unicl_losses = 0.0, 0.0
            correct = torch.tensor(0, device=self._device)
            total = torch.tensor(0, device=self._device)

            for i, batch in enumerate(train_loader):
                if isinstance(batch[1], dict) and 'stage_id'in batch[1]:
                    _, data_dict, targets = batch
                    inputs = data_dict['image'].to(self._device)
                    state_ids = data_dict['stage_id'].to(self._device)
                else:
                    _, inputs, targets = batch
                    state_ids = torch.full((inputs.size(0),), 4, dtype=torch.long).to(self._device)
                    inputs = inputs.to(self._device)
                targets = targets.to(self._device).long()

                # 1) 分类分支：forward_for_classification
                with torch.no_grad():
                    # 构造全类别文本（所有类别）
                    text_batch = [templates.format(lbl) for lbl in total_labels]
                    cls_logits = self.forward_for_classification(inputs, text_batch)
                ce_loss = torch.nn.functional.cross_entropy(cls_logits, targets)

                # 2) 三路对比分支：先为本批生成对应文本
                labels_string = [class_to_label[int(t)] for t in targets]
                batch_texts = [templates.format(lbl) for lbl in labels_string]  # 生成与每个样本对应的文本描述
                image_feats, text_feats, state_feats, proto_feats, logit_scale = \
                    self._network.forward_tri_modal(inputs, batch_texts, state_ids)

                # 额外计算 clip 对比损失，使用同一 batch 文本输入
                batch_text_features = self._network.encode_text(self._network.tokenizer(batch_texts).to(self._device), normalize=True)
                batch_text_features = torch.nn.functional.normalize(batch_text_features, dim=1)
                img_norm = torch.nn.functional.normalize(self._network.encode_image(inputs), dim=1)
                clip_loss_val = cliploss(img_norm, batch_text_features, logit_scale)

                # UniCL 三路对比损失（对 image_feats, text_feats, state_feats）
                evolution_embeddings = getattr(self._network, 'evolution_embeddings', None)
                unicl_val, _ = unicl_loss(
                    image_feats, text_feats, state_feats, 
                    targets, state_ids,
                    state_distance=self.state_distance, 
                    epoch=epoch, max_epoch=self.tuned_epoch,
                    evolution_features=evolution_embeddings  # 新增：传入演化特征
                )
                total_loss = ce_loss + clip_loss_val + 0.3 * unicl_val
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                losses += total_loss.item()
                unicl_losses += unicl_val.item()
                _, preds = torch.max(cls_logits, dim=1)
                correct += (preds == targets).sum()
                total += targets.size(0)
            scheduler.step()
            train_acc = np.around(correct.cpu().numpy() * 100 / total.cpu().numpy(), 2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = f"Task {self._cur_task}, Epoch {epoch+1}/{self.tuned_epoch} => " \
                   f"Loss {losses/len(train_loader):.3f}, UniCL {unicl_losses/len(train_loader):.3f}, " \
                   f"Train_acc {train_acc:.2f}, Test_acc {test_acc:.2f}"
            prog_bar.set_description(info)
            
            # 每隔一定epoch应用虫态演化
            if (epoch + 1) % 2 == 0:  # 例如每2个epoch进行一次演化
                logging.info(f"Epoch {epoch+1}: 应用虫态演化图网络更新原型...")
                evolution_embeddings = self._network.evolve_state_prototypes()
                self._network.evolution_embeddings = evolution_embeddings  # 缓存演化嵌入特征
                
                # 可选: 可视化当前虫态演化路径
                if (epoch + 1) == self.tuned_epoch:  # 最后一个epoch时可视化
                    self._visualize_evolution_paths()
        
            # 在每个epoch结束后:
        
            # 1. 先更新虫态演化网络
            if (epoch + 1) % 2 == 0:  # 例如每2个epoch更新一次
                logging.info(f"Epoch {epoch+1}: 应用虫态演化图网络更新原型...")
                evolution_embeddings = self._network.evolve_state_prototypes()
                self._network.evolution_embeddings = evolution_embeddings  # 缓存演化嵌入特征
                
                # 注释掉或删除原有的虫态距离矩阵更新代码
                # if hasattr(self, 'state_distance'):
                #     logging.info("更新虫态距离矩阵...")
                #     for class_id, state_dict in self._network.img_prototypes_by_state.items():
                #         for state_id1, proto1 in state_dict.items():
                #             for state_id2, proto2在 state_dict.items():
                #                 if state_id1 != state_id2:
                #                     # 计算原型间相似度，更新距离矩阵
                #                     sim = F.cosine_similarity(proto1.unsqueeze(0), proto2.unsqueeze(0))
                #                     dist = 1.0 - sim.item()  # 转换为距离
                #                     # 修改这里，尝试直接使用update_history而非update_distance
                #                     self.state_distance.update_history(state_id1, state_id2, dist, weight=0.1)

            # 在最后一个epoch后可视化
            if (epoch + 1) == self.tuned_epoch:
                self._visualize_evolution_paths()
                
            # 使用统一接口更新所有虫态相关信息
            evolution_results = self._network.state_evolution_graph.evolve_and_update(
                self._network.img_prototypes_by_state,
                epoch=epoch,
                max_epoch=self.tuned_epoch
            )

            # 解包结果
            evolved_prototypes = evolution_results['prototypes']
            evolution_embeddings = evolution_results['embeddings']
            distance_matrix = evolution_results['distances']

            # 更新网络状态
            self._network.img_prototypes_by_state = evolved_prototypes
            self._network.evolution_embeddings = evolution_embeddings

            # 更新虫态距离矩阵
            if hasattr(self, 'state_distance'):
                self.update_state_distance_matrix(self.data_manager)

        # 训练结束后，最后执行一次整合优化
        self._network.state_evolution_graph.integrate_with_state_distance(self.state_distance)
        self._network.evolve_state_prototypes()

    def forward_for_classification(self, images, text_list):
        """
        单独的分类前向：针对所有类别生成文本特征 (num_classes,D)，
        与图像特征 (batch_size,D) 做内积 -> [batch_size, num_classes]。
        text_list: 包含所有类别的描述字符串，长度等于 self._total_classes
        """
        # encode_image -> [B, D]
        image_features = self._network.encode_image(images)
        image_features = F.normalize(image_features, dim=1)

        with torch.no_grad():
            texts_tokenized = self._network.tokenizer(text_list).to(self._device)
            text_features = self._network.encode_text(texts_tokenized)  # [num_classes, D]
            text_features = F.normalize(text_features, dim=1)

        # 计算相似度 => logits: [B, num_classes]
        logits = image_features @ text_features.t()
        return logits

    @torch.no_grad()
    def _compute_accuracy(self, model, loader):
        """
        测试/评估时，只使用上面定义的 forward_for_classification，
        严格保证输出维度 [batch_size, num_classes]，去除多路特征零填充。
        """
        model.eval()
        correct, total = 0, 0
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]  # 只用一个模板做推理
        all_labels = class_to_label[:self._total_classes]

        for _, batch in enumerate(loader):
            if isinstance(batch[1], dict) and 'stage_id'in batch[1]:
                _, data_dict, targets = batch
                inputs = data_dict['image'].to(self._device)
            else:
                _, inputs, targets = batch
                inputs = inputs.to(self._device)

            targets = targets.long().to(self._device)

            # 准备完整类别文本
            text_list = [templates.format(lbl) for lbl in all_labels]
            logits = self.forward_for_classification(inputs, text_list)  # [B, num_classes]

            # 计算准确率
            _, preds = torch.max(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        return np.around((correct / total) * 100, decimals=2)

    def _eval_cnn(self, loader):
        """
        如果需要Top-K评估，也可使用 forward_for_classification。
        """
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]
        all_labels = class_to_label[:self._total_classes]
        y_pred, y_true = [], []

        for _, batch in enumerate(loader):
            if isinstance(batch[1], dict) and 'stage_id'in batch[1]:
                _, data_dict, targets = batch
                inputs = data_dict['image'].to(self._device)
            else:
                _, inputs, targets = batch
                inputs = inputs.to(self._device)

            targets = targets.long().to(self._device)
            text_list = [templates.format(lbl) for lbl in all_labels]
            logits = self.forward_for_classification(inputs, text_list)  # [B, num_classes]

            # 修改：确保topk不超过可用类别数
            k = min(self.topk, logits.size(1))  
            topk_preds = torch.topk(logits, k=k, dim=1)[1]
            
            # 如果k小于self.topk，需要填充结果保持一致的维度
            if k < self.topk:
                padding = torch.zeros(logits.size(0), self.topk - k, device=logits.device, dtype=torch.long)
                topk_preds = torch.cat([topk_preds, padding], dim=1)
                
            y_pred.append(topk_preds)
            y_true.append(targets)

        y_pred_tensor = torch.cat(y_pred, dim=0)
        y_true_tensor = torch.cat(y_true, dim=0)
        return y_pred_tensor.cpu().numpy(), y_true_tensor.cpu().numpy()

    def _visualize_evolution_paths(self):
        """可视化所有类别的虫态演化路径"""
        try:
            for class_id in range(self._total_classes):
                # 检查原型是否存在
                if (class_id in self._network.img_prototypes_by_state and 
                    len(self._network.img_prototypes_by_state[class_id]) >= 2):
                
                    # 获取该类的所有原型
                    proto_dict = self._network.img_prototypes_by_state[class_id]
                    proto_list = []
                
                    # 确保所有原型都是有效的Tensor
                    valid_protos = True
                    for state_id in sorted(proto_dict.keys()):
                        proto = proto_dict[state_id]
                        if proto is None:
                            valid_protos = False
                            print(f"警告: 类别 {class_id} 的状态 {state_id} 原型为None")
                            break
                        proto_list.append(proto)
                
                    # 只在所有原型都有效时可视化
                    if valid_protos and len(proto_list) >= 2:
                        self._network.state_evolution_graph.visualize_evolution_path(
                            class_id, 
                            proto_list,
                            save_dir=f"./results/task_{self._cur_task}/evolution"
                        )
        except Exception as e:
            print(f"可视化虫态演化路径时出错: {e}")
            import traceback
            traceback.print_exc()

    def update_state_distance_matrix(self, data_manager):
        """使用时序图卷积网络更新虫态距离矩阵"""
        if not hasattr(self, 'state_distance'):
            return
            
        with torch.no_grad():
            # 获取类别和状态信息
            if hasattr(self._network, 'state_evolution_graph'):
                # 使用时序图卷积网络获取演化结果
                evolution_results = self._network.state_evolution_graph.evolve_and_update(
                    self._network.img_prototypes_by_state
                )
                
                # 更新原型
                if 'prototypes' in evolution_results:
                    self._network.img_prototypes_by_state = evolution_results['prototypes']
                
                # 更新演化嵌入特征
                if 'embeddings' in evolution_results:
                    self._network.evolution_embeddings = evolution_results['embeddings']
                    
                # 更新距离矩阵
                if 'distances' in evolution_results:
                    state_distances = evolution_results['distances']
                    for s1 in state_distances:
                        for s2 in state_distances[s1]:
                            distance = state_distances[s1][s2]
                            # 更新距离矩阵
                            old_dist = self.state_distance.distance_factors[s1, s2].item()
                            weight = 0.3  # 时序模型的贡献权重
                            new_dist = (1-weight) * old_dist + weight * distance
                            self.state_distance.distance_factors[s1, s2] = new_dist
                            self.state_distance.distance_factors[s2, s1] = new_dist  # 保持对称性
                    
                    # 记录重要更新到历史记录
                    self.state_distance.update_history.append({
                        'epoch': len(self.state_distance.update_history),
                        'source': 'temporal_gcn',
                        'changes': len(state_distances)
                    })

from models.dynamic_modal_graph import DynamicRelationModeler, DynamicGCN, GlobalConnectivityExtractor

class DMIG_Net(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes):
        super().__init__()
        self.relation_modeler = DynamicRelationModeler(feature_dim, hidden_dim)
        self.graph_network = DynamicGCN(feature_dim, hidden_dim, feature_dim)
        self.global_extractor = GlobalConnectivityExtractor(feature_dim)
    
    def forward(self, inputs_dict, task_id=0):
        features_dict = {
            'image': self.encode_image(inputs_dict['image'], normalize=True),
            'text': self.encode_text(inputs_dict['text'], normalize=True),
            'state': self.encode_state(inputs_dict['state_ids'], normalize=True)
        }
        node_features, edge_index, edge_weights, node_types = self.relation_modeler(features_dict, task_id)
        updated_features = self.graph_network(node_features, edge_index, edge_weights)
        global_features = self.global_extractor(updated_features)
        return global_features


