
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def analyze_state_evolution(model, class_to_label, save_dir="./results/analysis"):
    """分析并可视化虫态原型的演化效果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 收集所有虫态原型
    all_protos = []
    all_labels = []
    all_states = []
    all_texts = []
    
    state_names = {1: "Larva", 3: "Nymph", 4: "Adult"}
    
    for class_id, state_dict in model.img_prototypes_by_state.items():
        class_name = class_to_label[class_id]
        for state_id, proto in state_dict.items():
            all_protos.append(proto.cpu().detach())
            all_labels.append(class_id)
            all_states.append(state_id)
            state_name = state_names.get(state_id, f"未知虫态{state_id}")
            all_texts.append(f"{class_name} ({state_name})")
    
    if not all_protos:
        print("没有虫态原型可供分析")
        return
        
    all_protos = torch.stack(all_protos).numpy()
    
    # 使用t-SNE将原型降到2D进行可视化
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(all_protos)-1)))
    protos_2d = tsne.fit_transform(all_protos)
    
    # 按类别和虫态类型可视化
    plt.figure(figsize=(12, 10))
    
    # 颜色映射
    unique_classes = np.unique(all_labels)
    class_cmap = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_colors = {cls: class_cmap[i] for i, cls in enumerate(unique_classes)}
    
    # 虫态映射到标记形状
    state_markers = {1: 'o', 3: '^', 4: 's'}
    state_sizes = {1: 100, 3: 100, 4: 100}
    
    # 绘制每个原型
    for i, (proto_2d, class_id, state_id, text) in enumerate(zip(protos_2d, all_labels, all_states, all_texts)):
        color = class_colors.get(class_id, 'gray')
        marker = state_markers.get(state_id, 'o')
        size = state_sizes.get(state_id, 100)
        
        plt.scatter(
            proto_2d[0], proto_2d[1],
            c=[color],
            marker=marker,
            s=size,
            alpha=0.7,
            edgecolors='black',
            linewidths=1
        )
        
        plt.annotate(
            text,
            (proto_2d[0], proto_2d[1]),
            textcoords="offset points",
            xytext=(0, 5),
            ha='center',
            fontsize=8
        )
    
    # 连接同一类别的不同虫态
    for class_id in unique_classes:
        class_indices = [i for i, c in enumerate(all_labels) if c == class_id]
        if len(class_indices) <= 1:
            continue
            
        class_states = [all_states[i] for i in class_indices]
        
        # 检查是否有演化路径
        if 1 in class_states and 4 in class_states:  # 幼虫→成虫
            idx1 = class_indices[class_states.index(1)]
            idx4 = class_indices[class_states.index(4)]
            plt.arrow(
                protos_2d[idx1, 0], protos_2d[idx1, 1],
                protos_2d[idx4, 0] - protos_2d[idx1, 0],
                protos_2d[idx4, 1] - protos_2d[idx1, 1],
                head_width=0.3, head_length=0.5,
                fc=class_colors[class_id], ec=class_colors[class_id],
                alpha=0.6
            )
            
        if 3 in class_states and 4 in class_states:  # 若虫→成虫
            idx3 = class_indices[class_states.index(3)]
            idx4 = class_indices[class_states.index(4)]
            plt.arrow(
                protos_2d[idx3, 0], protos_2d[idx3, 1],
                protos_2d[idx4, 0] - protos_2d[idx3, 0],
                protos_2d[idx4, 1] - protos_2d[idx3, 1],
                head_width=0.3, head_length=0.5,
                fc=class_colors[class_id], ec=class_colors[class_id],
                alpha=0.6
            )
    
    class_legend = []
    for cls in unique_classes:
        class_name = class_to_label[cls]
        class_patch = plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=class_colors[cls],
            markersize=10,
            label=class_name
        )
        class_legend.append(class_patch)
    
    # 虫态图例
    state_legend = []
    for state_id, marker in state_markers.items():
        if state_id in state_names:
            state_patch = plt.Line2D(
                [0], [0],
                marker=marker,
                color='k',
                markersize=10,
                label=state_names[state_id]
            )
            state_legend.append(state_patch)
    
    # 创建两个图例
    plt.legend(handles=class_legend, title="昆虫类别", loc="upper left")
    plt.legend(handles=state_legend, title="虫态类型", loc="lower right")
    
    plt.title("虫态原型特征分布与演化路径", fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 保存图像
    save_path = os.path.join(save_dir, "state_evolution_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存虫态演化分析结果: {save_path}")