import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCIFAR224, \
    iImageNetR,iImageNetA,CUB, objectnet, omnibenchmark, vtab, Caltech101, Food101, Flowers, \
    Aircraft,UCF101,StanfordCars, SUN, IIMinsects202
import json
import os

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):

        # load class to label name json file
        with open('./utils/labels.json', 'r') as f:
            self._class_to_label = json.load(f)[dataset_name]
        print(self._class_to_label)
        with open('./utils/templates.json', 'r', encoding="utf-8") as f:  # 修改后使用 utf-8 编码打开文件
            self._data_to_prompt = json.load(f)[dataset_name]
        print(self._data_to_prompt)
        
        
        self.dataset_name = dataset_name
        # 添加对idata的引用保存
        self._setup_data(dataset_name, shuffle, seed)
        # assert init_cls <= len(self._class_order), "No enough classes."
        if init_cls > len(self._class_order):
            print("No enough classes.")
            self._increments=[len(self._class_order)]
        else:
            self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)
        print('Training class stages:',self._increments )
        

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0 ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1 )
            val_indx = np.random.choice( len(class_data), val_samples_per_class, replace=False)
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select( appendent_data, appendent_targets, low_range=idx, high_range=idx + 1)
                val_indx = np.random.choice( len(append_data), val_samples_per_class, replace=False)
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate( train_targets )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(train_data, train_targets, trsf, self.use_path), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        # 获取并保存idata引用
        self.idata = _get_idata(dataset_name)
        
        # 确保数据只加载一次
        if not hasattr(self.idata, '_data_loaded') or not self.idata._data_loaded:
            print(f"初始化{dataset_name}数据集...")
            self.idata.download_data()
        else:
            print(f"使用已加载的{dataset_name}数据集")

        # 获取基本数据
        self._train_data, self._train_targets = self.idata.train_data, self.idata.train_targets
        self._test_data, self._test_targets = self.idata.test_data, self.idata.test_targets
        self.use_path = self.idata.use_path

        # 获取转换函数
        self._train_trsf = self.idata.train_trsf
        self._test_trsf = self.idata.test_trsf
        self._common_trsf = self.idata.common_trsf

        # 设置类别顺序
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = self.idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # 映射类别索引
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

        # 更新标签
        _class_to_label=[self._class_to_label[i] for i in self._class_order]
        self._class_to_label = _class_to_label
        print('After shuffle, class_to_label is: ', self._class_to_label)


    def _select(self, x, y, low_range, high_range):
        # 如果 x、y 不是 numpy 数组则转换
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        idxes = np.where((y >= low_range) & (y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))

    def get_multimodal_dataset(self, indices, source, mode, appendent=None):
        """返回支持三路投影的多模态数据集"""
        base_dataset = self.get_dataset(indices, source, mode, appendent)
        
        # 保存对idata的引用，方便后续访问
        self.idata = _get_idata(self.dataset_name)
        
        return InsectsMultiModalDataset(base_dataset, self)

    def get_stage_prompt(self, class_idx, stage_id):
        """获取结合类别和虫态的提示模板"""
        class_name = self._class_to_label[class_idx]
        
        # 获取虫态名称
        idata = _get_idata(self.dataset_name)
        stage_name = idata.get_stage_description(stage_id)
        
        # 从templates中选择一个随机提示并格式化
        template = np.random.choice(self._data_to_prompt)
        # 替换模板中的{类别}为类别名称
        prompt = template.replace("{类别}", class_name)
        # 添加虫态信息
        prompt = prompt.replace("{虫态}", stage_name)
        
        return prompt


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))

def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar224":
        return iCIFAR224()
    elif name== "imagenetr":
        return iImageNetR()
    elif name=="imageneta":
        return iImageNetA()
    elif name=="objectnet":
        return objectnet()
    elif name=="cub":
        return CUB()
    elif name=="caltech101":
        return Caltech101()
    elif name=="food101":
        return Food101()
    elif name=="flowers":
        return Flowers()
    elif name=="aircraft":
        return Aircraft()
    elif name=="ucf101":
        return UCF101()
    elif name=="cars":
        return StanfordCars()
    elif name=="sun":
        return SUN()
    elif name == "iiminsects202":
        return IIMinsects202()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def _get_idata_image_only(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name== "cifar224":
        return iCIFAR224()
    elif name== "imagenetr":
        return iImageNetR()
    elif name=="imageneta":
        return iImageNetA()
    elif name=="cub":
        return CUB()
    elif name=="objectnet":
        return objectnet()
    elif name=="omnibenchmark":
        return omnibenchmark()
    elif name=="vtab":
        return vtab()
    elif name == "iiminsects202":
        return IIMinsects202()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    try:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    except Exception as e:
        print(f"[ERROR] 无法打开图片：{path}, 异常信息：{e}")
        raise e



# def accimage_loader(path):
#     """
#     Ref:
#     https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
#     accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
#     accimage is available on conda-forge.
#     """
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


# def default_loader(path):
#     """
#     Ref:
#     https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
#     """
#     from torchvision import get_image_backend

#     if get_image_backend() == "accimage":
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)

class LaionData(Dataset):
    def __init__(self, txt_path):
        self.transform = transforms.Compose([
            transforms.Resize((224,224),transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        self.img_list = [line.split()[0] for line in lines]
        self.txt_list = [line.split()[1] for line in lines]

    def __getitem__(self, index):
        txt_path = self.txt_list[index]
        img = Image.open(self.img_list[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        with open(txt_path, 'r') as f:
            txt = f.read().strip()
        return img, txt

    def __len__(self):
        return len(self.img_list)

class InsectsMultiModalDataset(Dataset):
    """支持三路投影的昆虫多模态数据集类"""
    def __init__(self, dataset, data_manager):
        self.dataset = dataset  # 原始数据集
        self.data_manager = data_manager
        
        # 使用data_manager中的idata引用，避免重复实例化
        self.insect_data = data_manager.idata
        
        # 确保数据已加载
        if not hasattr(self.insect_data, '_data_loaded') or not self.insect_data._data_loaded:
            print("确保数据已正确加载...")
            self.insect_data.download_data()
        
        # 验证虫态信息是否存在
        if not hasattr(self.insect_data, 'train_stages') or not hasattr(self.insect_data, 'test_stages'):
            print("警告: 无法找到虫态信息，尝试从路径提取...")
            self._extract_stage_info_from_paths()
            
            # 如果仍然没有虫态信息，则报错
            if (not hasattr(self.insect_data, 'train_stages') or len(self.insect_data.train_stages) == 0) and \
               (not hasattr(self.insect_data, 'test_stages') or len(self.insect_data.test_stages) == 0):
                raise ValueError("无法获取虫态信息，请检查数据集结构")
        
        # 构建索引映射，只需要做一次
        self._build_index_mapping()
    
    def _extract_stage_info_from_paths(self):
        """从图像路径中提取虫态信息"""
        # 避免重复执行
        if hasattr(self, '_extracted_stages') and self._extracted_stages:
            return
            
        # 确认具有图像路径数据
        if not hasattr(self.insect_data, 'train_data') or not hasattr(self.insect_data, 'test_data'):
            print("ERROR: 数据集缺少必要属性")
            return
            
        # 初始化虫态数组(如果不存在)
        if not hasattr(self.insect_data, 'train_stages') or self.insect_data.train_stages is None:
            self.insect_data.train_stages = np.zeros(len(self.insect_data.train_data), dtype=np.int32)
        if not hasattr(self.insect_data, 'test_stages') or self.insect_data.test_stages is None:
            self.insect_data.test_stages = np.zeros(len(self.insect_data.test_data), dtype=np.int32)

        # 从路径中提取训练集虫态
        print("从训练集路径中提取虫态信息...")
        for i, path in enumerate(self.insect_data.train_data):
            try:
                # 标准化路径格式并提取虫态
                parts = os.path.normpath(path).replace('\\', '/').split('/')
                stage_part = None
                for part in reversed(parts):  # 从后往前找
                    if part.isdigit():
                        stage_part = part
                        break
                
                if stage_part is not None:
                    self.insect_data.train_stages[i] = int(stage_part)
                else:
                    # 回退到正则表达式提取方式
                    import re
                    # 查找形式为 /数字/ 的部分
                    match = re.search(r'/(\d+)/', path.replace('\\', '/'))
                    if match:
                        self.insect_data.train_stages[i] = int(match.group(1))
                    else:
                        self.insect_data.train_stages[i] = 4  # 默认成虫
            except Exception as e:
                # 出错时设为默认值
                self.insect_data.train_stages[i] = 4
        
        # 使用相同方法处理测试集
        print("从测试集路径中提取虫态信息...")
        for i, path in enumerate(self.insect_data.test_data):
            try:
                parts = os.path.normpath(path).replace('\\', '/').split('/')
                stage_part = None
                for part in reversed(parts):
                    if part.isdigit():
                        stage_part = part
                        break
                        
                if stage_part is not None:
                    self.insect_data.test_stages[i] = int(stage_part)
                else:
                    import re
                    match = re.search(r'/(\d+)/', path.replace('\\', '/'))
                    if match:
                        self.insect_data.test_stages[i] = int(match.group(1))
                    else:
                        self.insect_data.test_stages[i] = 4
            except Exception:
                self.insect_data.test_stages[i] = 4
                
        # 标记已提取
        self._extracted_stages = True
    
    def _build_index_mapping(self):
        """构建图像路径到原始数据索引的映射"""
        # 避免重复构建
        if hasattr(self, 'path_to_idx_train'):
            return
            
        self.path_to_idx_train = {}
        self.path_to_idx_test = {}
        self.basename_to_idx_train = {}
        self.basename_to_idx_test = {}
        
        # 如果数据集使用路径
        if hasattr(self.insect_data, 'use_path') and self.insect_data.use_path:
            # 构建训练集映射
            for i, path in enumerate(self.insect_data.train_data):
                norm_path = os.path.normpath(path).replace('\\', '/').lower()
                self.path_to_idx_train[norm_path] = i
                
                # 文件名映射 (不带路径)
                basename = os.path.basename(path).lower()
                self.basename_to_idx_train[basename] = i
            
            # 构建测试集映射
            for i, path in enumerate(self.insect_data.test_data):
                norm_path = os.path.normpath(path).replace('\\', '/').lower()
                self.path_to_idx_test[norm_path] = i
                
                basename = os.path.basename(path).lower()  
                self.basename_to_idx_test[basename] = i
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """获取图像、虫态和标签"""
        orig_idx, image, label = self.dataset[idx]
        
        # 尝试获取虫态ID
        stage_id = 4  # 默认成虫
        
        try:
            # 获取原始图像路径
            if hasattr(self.dataset, 'images'):
                img_path = self.dataset.images[idx]
                
                # 标准化路径
                norm_path = os.path.normpath(img_path).replace('\\', '/').lower()
                basename = os.path.basename(img_path).lower()
                
                # 依次尝试不同索引匹配方法
                if norm_path in self.path_to_idx_train:
                    orig_idx = self.path_to_idx_train[norm_path]
                    stage_id = self.insect_data.train_stages[orig_idx]
                elif norm_path in self.path_to_idx_test:  
                    orig_idx = self.path_to_idx_test[norm_path]
                    stage_id = self.insect_data.test_stages[orig_idx]
                elif basename in self.basename_to_idx_train:
                    orig_idx = self.basename_to_idx_train[basename]
                    stage_id = self.insect_data.train_stages[orig_idx]
                elif basename in self.basename_to_idx_test:
                    orig_idx = self.basename_to_idx_test[basename]
                    stage_id = self.insect_data.test_stages[orig_idx]
                else:
                    # 直接从路径提取
                    parts = norm_path.split('/')
                    for part in parts:
                        if part.isdigit():
                            stage_id = int(part)
                            break
        except Exception as e:
            print(f"虫态识别失败: {e}, 使用默认值4(成虫)")
        
        # 创建包含所有信息的字典
        data_dict = {
            'image': image,
            'label': label,
            'stage_id': stage_id,
            'orig_idx': orig_idx
        }
        
        return orig_idx, data_dict, label