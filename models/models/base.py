import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist

EPSILON = 1e-8
batch_size = 128


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 4

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    


    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        
        # 健壮的 top-k 计算
        try:
            correct = 0
            for i in range(len(y_true)):
                if y_true[i] in y_pred[i, :self.topk]:
                    correct += 1
            ret[f"top{self.topk}"] = np.around((correct * 100.0) / len(y_true), decimals=2)
        except Exception as e:
            print(f"计算top{self.topk}准确率时出错: {str(e)}")
            ret[f"top{self.topk}"] = 0.0
        
        return ret
    
    def _evaluate_zs(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._total_classes) # indx< total are old classes, >= are new unseen classes.
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around((y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),decimals=2)
        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        if self.args["convnet_type"].lower()!="clip" or self.args["model_name"].lower()=="l2p" or self.args["model_name"].lower()=="dualprompt":
            return cnn_accy, nme_accy, None, None, None, None
        else:
            y_pred, y_true = self._eval_zero_shot()
            zs_acc= self._evaluate_zs(y_pred, y_true)
            zs_seen, zs_unseen, zs_harmonic, zs_total = zs_acc["grouped"]["old"], zs_acc["grouped"]["new"], zs_acc["grouped"]["harmonic"], zs_acc["grouped"]["total"]

        return cnn_accy, nme_accy, zs_seen, zs_unseen, zs_harmonic, zs_total

    def _eval_zero_shot(self):  
        self._network.eval()
        class_to_label=self.data_manager._class_to_label
        templates=self.data_manager._data_to_prompt
        total_labels=class_to_label  # 所有类别
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0).to(self._device)

        test_dataset = self.data_manager.get_dataset(np.arange(0, len(total_labels)), source="test", mode="test" )
        loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)  # 将标签也移到GPU
            with torch.no_grad():
                image_features=self._network.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                outputs= image_features @ text_features.T
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  
            y_pred.append(predicts)
            y_true.append(targets)
        
        # 在GPU上合并结果
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        
        # 最后一步转为CPU
        return y_pred.cpu().numpy(), y_true.cpu().numpy()

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)


    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)  # 将targets移到GPU
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts == targets).sum()  # 保持在GPU上计算
            total += len(targets)

        return np.around((correct.item() / total) * 100, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, _inputs, _targets in loader:
            # 处理多种输入格式
            if isinstance(_inputs, dict) and 'image' in _inputs:
                _inputs = _inputs['image'].to(self._device)
            elif isinstance(_inputs, dict) and 'stage_id' in _inputs:
                data_dict = _inputs
                _inputs = data_dict['image'].to(self._device)
            else:
                _inputs = _inputs.to(self._device)
                
            _targets = _targets.to(self._device)
            _preds = torch.argmax(self._network(_inputs)['logits'], dim=1)
            y_pred.append(_preds.cpu().numpy())
            y_true.append(_targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)
    

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            # 处理多种输入格式 - 新增以下代码
            if isinstance(_inputs, dict) and 'image' in _inputs:
                # 如果输入是包含'image'键的字典
                _inputs = _inputs['image'].to(self._device)
            elif isinstance(_inputs, dict) and 'stage_id' in _inputs:
                # 处理包含虫态信息的字典
                data_dict = _inputs
                _inputs = data_dict['image'].to(self._device)
            else:
                # 常规张量输入
                _inputs = _inputs.to(self._device)
                
            _targets = _targets.numpy()
            _vectors = tensor2numpy(self._network.extract_vector(_inputs))
            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info("构建新类别的exemplars，每类最多{}个样本".format(m))
        
        # 初始化exemplar内存结构
        if not hasattr(self, "_data_memory_by_state"):
            self._data_memory_by_state = {}  # {类别: {虫态: [样本列表]}}
            self._targets_memory_by_state = {}  # {类别: {虫态: [标签列表]}}
        
        # 计算旧类别的均值向量
        _class_means = np.zeros((self._total_classes, self.feature_dim))
        
        # 处理旧类别
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = self._data_memory[mask], self._targets_memory[mask]
            
            # 加载数据集
            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", 
                appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            
            # 提取特征
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            _class_means[class_idx, :] = mean
        
        # 为新类别构建exemplars并计算均值
        for class_idx in range(self._known_classes, self._total_classes):
            # 获取类别数据
            data, targets, class_dset = data_manager.get_multimodal_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            
            # 提取向量
            vectors, _, state_ids = self._extract_vectors_with_states(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)
            
            # 按虫态分组
            unique_states = np.unique(state_ids)
            samples_per_state = max(m // len(unique_states), 1)  # 每个虫态至少1个样本
            
            # 初始化类别的虫态字典
            if class_idx not in self._data_memory_by_state:
                self._data_memory_by_state[class_idx] = {}
                self._targets_memory_by_state[class_idx] = {}
            
            # 为每个虫态选择exemplars
            for state_id in unique_states:
                state_mask = (state_ids == state_id)
                state_vectors = vectors[state_mask]
                state_data = data[state_mask]
                
                # 如果该虫态样本不足，全部保留
                if len(state_vectors) <= samples_per_state:
                    selected_exemplars = state_data
                    exemplar_targets = np.full(len(selected_exemplars), class_idx)
                else:
                    # Herding算法选择样本
                    selected_exemplars = []
                    exemplar_vectors = []
                    state_class_mean = np.mean(state_vectors, axis=0)
                    
                    for k in range(1, samples_per_state + 1):
                        S = np.sum(exemplar_vectors, axis=0)
                        mu_p = (state_vectors + S) / k
                        i = np.argmin(np.sqrt(np.sum((state_class_mean - mu_p) ** 2, axis=1)))
                        
                        selected_exemplars.append(np.array(state_data[i]))
                        exemplar_vectors.append(np.array(state_vectors[i]))
                        
                        # 删除已选样本
                        state_vectors = np.delete(state_vectors, i, axis=0)
                        state_data = np.delete(state_data, i, axis=0)
                    
                    selected_exemplars = np.array(selected_exemplars)
                    exemplar_targets = np.full(len(selected_exemplars), class_idx)
                
                # 存储到虫态exemplar内存
                self._data_memory_by_state[class_idx][state_id] = selected_exemplars
                self._targets_memory_by_state[class_idx][state_id] = exemplar_targets
                
                # 同时添加到主exemplar内存
                self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) > 0 else selected_exemplars
                self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if len(self._targets_memory) > 0 else exemplar_targets
            
            # 计算该类别的全局原型
            exemplar_dset = data_manager.get_multimodal_dataset(
                [], source="train", mode="test",
                appendent=(self._data_memory_by_state[class_idx], self._targets_memory_by_state[class_idx])
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            
            _class_means[class_idx, :] = mean
        
        self._class_means = _class_means
