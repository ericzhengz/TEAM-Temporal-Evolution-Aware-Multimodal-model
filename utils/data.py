import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import os
import glob

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


def build_transform_vit(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    return t

def build_transform(is_train, args):
    input_size = 224
    t=[  
        transforms.Resize((224,224),transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
    return t

class iCIFAR224(iData):
    use_path = False

    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [ ]
    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("/data/sunhl/data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("/data/sunhl/data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]


    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/data/sunhl/data/imagenet-r/train"
        test_dir = "/data/sunhl/data/imagenet-r/test"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        # print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "../pcil/data/imagenet-a/train/"
        test_dir = "../pcil/data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        # print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class objectnet(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/data/sunhl/data/objectnet/train/"
        test_dir = "/data/sunhl/data/objectnet/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class CUB(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/data/sunhl/data/cub/train/"
        test_dir = "/data/sunhl/data/cub/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class Caltech101(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [  ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "../data/clip-data/caltech101/train/"
        test_dir = "..//data/clip-data/caltech101/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class Food101(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [  ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/data/sunhl/data/food101/train/"
        test_dir = "/data/sunhl/data/food101/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class Flowers(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [  ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "../data/clip-data/flowers/train/"
        test_dir = "..//data/clip-data/flowers/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class Aircraft(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [  ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/data/sunhl/data/aircraft/train/"
        test_dir = "/data/sunhl/data/aircraft/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class UCF101(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [  ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/data/sunhl/data/ucf/train/"
        test_dir = "/data/sunhl/data/ucf/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class StanfordCars(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [  ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/data/sunhl/data/cars/train/"
        test_dir = "/data/sunhl/data/cars/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class SUN(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [  ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/user/sunhl/Dataset/sun/train/"
        test_dir = "/user/sunhl/Dataset/sun/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class omnibenchmark(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/omnibenchmark/train/"
        test_dir = "./data/omnibenchmark/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class vtab(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/vtab-cil/vtab/train/"
        test_dir = "./data/vtab-cil/vtab/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

import json
class IIMinsects202(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    
    # 记录数据是否已加载的标志
    _data_loaded = False

    # 20个昆虫类别
    class_order = np.arange(20).tolist()  

    def download_data(self):
        # 避免重复加载
        if self._data_loaded:
            print("数据集已加载，跳过重复加载过程...")
            return
            
        # 数据集路径
        train_dir = "C:/Users/ASUS/Desktop/PROOF-main/data/IIMinsects202/train"  
        test_dir = "C:/Users/ASUS/Desktop/PROOF-main/data/IIMinsects202/test"
        
        # 加载虫态映射信息
        with open('./utils/templates.json', 'r', encoding="utf-8") as f:
            templates_data = json.load(f)
            self.state_mapping = templates_data.get('states', {})
        
        # 加载数据集并计时
        import time
        start_time = time.time()
        print("开始加载训练数据...")
        self.train_data, self.train_targets, train_class_map, self.train_stages = self.load_nested_dataset(train_dir)
        print(f"训练数据加载完成, 用时: {time.time() - start_time:.2f}秒")
        
        start_time = time.time()
        print("开始加载测试数据...")
        self.test_data, self.test_targets, _, self.test_stages = self.load_nested_dataset(test_dir, class_map=train_class_map)
        print(f"测试数据加载完成, 用时: {time.time() - start_time:.2f}秒")
        
        # 记录类别映射和数据集信息
        self.class_map = train_class_map
        self.inverse_class_map = {v: k for k, v in train_class_map.items()}
        
        # 数据加载完成标记
        self._data_loaded = True
        
        # 输出更详细的数据集信息
        print(f"数据集总计: {len(self.train_data) + len(self.test_data)} 图像")
        print(f"训练集: {len(self.train_data)} 图像, 测试集: {len(self.test_data)} 图像")
        
        # 输出虫态统计
        self._print_stage_statistics()
    
    def _print_stage_statistics(self):
        """输出各虫态的样本统计"""
        # 统计训练集
        train_stages = np.array(self.train_stages)
        unique_train_stages = np.unique(train_stages)
        for stage_id in sorted(unique_train_stages):
            stage_name = self.get_stage_description(stage_id)
            count = np.sum(train_stages == stage_id)
            print(f"虫态 {stage_id} ({stage_name}): 训练集 {count} 图像", end="")
            
            # 统计测试集同一虫态
            test_stages = np.array(self.test_stages)
            if stage_id in test_stages:
                test_count = np.sum(test_stages == stage_id)
                print(f", 测试集 {test_count} 图像")
            else:
                print(", 测试集 0 图像")

    def load_nested_dataset(self, root_dir, class_map=None):
        """加载三层嵌套结构的数据集: 类别/虫态/图像"""
        images, targets, stages = [], [], []
        if class_map is None:
            class_map = {}
        
        # 记录加载统计
        num_images_loaded = 0
        num_classes_loaded = 0
        num_stages_found = set()
        
        # 遍历类别文件夹
        for class_folder in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_folder)
            # 跳过非文件夹或隐藏文件夹
            if not os.path.isdir(class_path) or class_folder.startswith('.'):
                continue
                
            # 分配类别ID
            if class_folder not in class_map:
                class_map[class_folder] = len(class_map)
            class_label = class_map[class_folder]
            num_classes_loaded += 1
            
            # 遍历虫态文件夹
            for stage_folder in sorted(os.listdir(class_path), key=lambda x: int(x) if x.isdigit() else float('inf')):
                stage_path = os.path.join(class_path, stage_folder)
                if not os.path.isdir(stage_path) or stage_folder.startswith('.'):
                    continue
                
                try:
                    stage_id = int(stage_folder)
                    num_stages_found.add(stage_id)
                    
                    # 获取该虫态下所有图像
                    for img_ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                        img_files = glob.glob(os.path.join(stage_path, img_ext))
                        for img_path in img_files:
                            if os.path.isfile(img_path) and not os.path.basename(img_path).startswith('.'):
                                images.append(img_path)
                                targets.append(class_label)
                                stages.append(stage_id)
                                num_images_loaded += 1
                                
                except ValueError:
                    print(f"警告: 跳过无效虫态文件夹: {stage_folder}")
        
        # 打印加载统计
        print(f"加载了 {num_images_loaded} 张图像, {num_classes_loaded} 个类别, {len(num_stages_found)} 个虫态")
        for stage_id in sorted(num_stages_found):
            stage_count = stages.count(stage_id)
            print(f"虫态 {stage_id}: {stage_count} 张图像")
            
        return images, np.array(targets), class_map, np.array(stages)

    def get_stage_description(self, stage_id):
        """获取虫态描述"""
        # 映射虫态ID到描述
        stage_descriptions = {
            1: "幼虫", 
            2: "蛹",
            3: "若虫",
            4: "成虫", 
            5: "虫卵"
        }
        return stage_descriptions.get(stage_id, "未知虫态")



