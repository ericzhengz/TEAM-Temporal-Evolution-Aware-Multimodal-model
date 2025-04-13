import copy
import logging
import torch
from torch import nn
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear
import timm
import torch.nn.functional as F
from convs.projections import Proj_Pure_MLP, MultiHeadAttention
from models.state_evolution import InsectLifecycleModel
from utils.toolkit import get_attribute

def get_convnet(args, pretrained=False):
    backbone_name = args["convnet_type"].lower()
    algorithm_name = args["model_name"].lower()
    if 'clip' in backbone_name:
        print('Using CLIP model as the backbone')
        import open_clip
        if backbone_name == 'clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name=='clip_laion2b':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name=='openai_clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        else:
            raise NotImplementedError("Unknown type {}".format(backbone_name))
    else:
        raise NotImplementedError("Unknown type {}".format(backbone_name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.convnet = get_convnet(args, pretrained)
        self.fc = None
        self.device = args["device"][0]
        self.to(self.device)

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features,
            'logits': logits
        }
        """
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights, gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)


class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            fc = SplitCosineLinear(in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy)
        return fc


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = self.alpha * x[:, low_range:high_range] + self.beta
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, args, pretrained, bias_correction=False):
        super().__init__(args, pretrained)
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        if self.bias_correction:
            logits = out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(logits, sum(self.task_sizes[:i]), sum(self.task_sizes[:i + 1]))
            out["logits"] = logits
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())
        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.convnet, self.preprocess, self.tokenizer = get_convnet(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def encode_image(self, x):
        return self.convnet.encode_image(x)
    
    def encode_text(self, x):
        return self.convnet.encode_text(x)
        
    def forward(self, x):
        x = self.convnet.encode_image(x)
        out = self.fc(x)
        return out


class SimpleClipNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.convnet, self.preprocess, self.tokenizer = get_convnet(args, pretrained)
        self.class_name = 'SimpleClipNet'
        self.args = args

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def encode_image(self, x):
        return self.convnet.encode_image(x)
    
    def encode_text(self, x):
        return self.convnet.encode_text(x)

    def forward(self, img, text):
        image_features, text_features, logit_scale = self.convnet(img, text)
        return image_features, text_features, logit_scale

    def re_initiate(self):
        print('re-initiate model')
        self.convnet, self.preprocess, self.tokenizer = get_convnet(self.args, True)


class Proof_Net(SimpleClipNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.projs_img = nn.ModuleList()
        self.projs_text = nn.ModuleList()
        self.projs_state = nn.ModuleList()  # 添加虫态投影列表
        self.args = args
        self._device = args["device"][0]
        self.projtype = get_attribute(self.args, 'projection_type', 'mlp')
        self.context_prompt_length_per_task = get_attribute(self.args, 'context_prompt_length_per_task', 3)
        
        self.sel_attn = MultiHeadAttention(1, self.feature_dim, self.feature_dim, self.feature_dim, dropout=0.1)
        self.img_prototypes = None
        self.context_prompts = nn.ParameterList()
        
        # 统一命名：使用 state_embedder 作为虫态嵌入模块
        self.state_embedder = InsectLifecycleModel(
            feature_dim=self.feature_dim,
            hidden_dim=self.feature_dim // 2,
            num_states=10  # 支持的最大虫态数
        ).to(self._device)

        # 为了简化接口，统一使用 state_evolution_graph 作为虫态演化模块名称
        self.state_evolution_graph = self.state_embedder
        
        # 初始化原型存储结构
        self.img_prototypes_by_state = {}
        self.evolution_embeddings = None

    def update_prototype(self, nb_classes):
        # 将一维原型结构改为二维结构
        if not hasattr(self, "img_prototypes_by_state"):
            self.img_prototypes_by_state = {}
        
        for class_id in range(nb_classes):
            if class_id not in self.img_prototypes_by_state:
                self.img_prototypes_by_state[class_id] = {}
        
        if self.img_prototypes is not None:
            nb_output = len(self.img_prototypes)
            self.img_prototypes = torch.cat([
                copy.deepcopy(self.img_prototypes).to(self._device),
                torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)
            ]).to(self._device)
        else:
            self.img_prototypes = torch.zeros(nb_classes, self.feature_dim).to(self._device)
        
        print(f'更新原型，现有 {nb_classes} 个类别原型和虫态原型字典')
    
    def update_context_prompt(self):
        for i in range(len(self.context_prompts)):
            self.context_prompts[i].requires_grad = False
        self.context_prompts.append(nn.Parameter(torch.randn(self.context_prompt_length_per_task, self.feature_dim).to(self._device)))
        print('update context prompt, now we have {} context prompts'.format(len(self.context_prompts) * self.context_prompt_length_per_task))
        self.context_prompts.to(self._device)
    
    def get_context_prompts(self):
        return torch.cat([item for item in self.context_prompts], dim=0)

    def encode_image(self, x, normalize: bool = False):
        x = x.to(self._device)
        basic_img_features = self.convnet.encode_image(x)
        img_features = [proj(basic_img_features) for proj in self.projs_img]
        img_features = torch.stack(img_features, dim=1)  # [bs, num_proj, dim]
        image_feas = torch.sum(img_features, dim=1)  # [bs, dim]
        return F.normalize(image_feas, dim=-1) if normalize else image_feas
        
    def encode_text(self, x, normalize: bool = False):
        x = x.to(self._device)
        basic_text_features = self.convnet.encode_text(x)
        text_features = [proj(basic_text_features) for proj in self.projs_text]
        text_features = torch.stack(text_features, dim=1)
        text_feas = torch.sum(text_features, dim=1)  # [bs, dim]
        return F.normalize(text_feas, dim=-1) if normalize else text_feas
        
    def encode_prototpyes(self, normalize: bool = False):
        self.img_prototypes = self.img_prototypes.to(self._device)
        img_features = [proj(self.img_prototypes) for proj in self.projs_img]
        img_features = torch.stack(img_features, dim=1)  # [nb_class, num_proj, dim]
        image_feas = torch.sum(img_features, dim=1)  # [nb_class, dim]
        return F.normalize(image_feas, dim=-1) if normalize else image_feas

    def extend_task(self):
        self.projs_img.append(self.extend_item())
        self.projs_text.append(self.extend_item())
        self.projs_state.append(self.extend_item())  # 为虫态添加新投影
        print(f"任务扩展: 添加新投影，当前共有 {len(self.projs_img)} 组三路投影")

    def extend_item(self):
        if self.projtype == 'pure_mlp':
            return Proj_Pure_MLP(self.feature_dim, self.feature_dim, self.feature_dim).to(self._device)
        else:
            raise NotImplementedError
    
    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)  # [bs, dim]
        text_features = self.encode_text(text, normalize=True)  # [bs, dim]
        prototype_features = self.encode_prototpyes(normalize=True)  # [nb_class, dim]
        context_prompts = self.get_context_prompts()  # [num_prompt, dim]

        len_texts = text_features.shape[0]
        len_protos = prototype_features.shape[0]
        len_context_prompts = context_prompts.shape[0]
        image_features = image_features.view(image_features.shape[0], -1, self.feature_dim)  # [bs, 1, dim]
        text_features = text_features.view(text_features.shape[0], self.feature_dim)  # [num_text, dim]
        prototype_features = prototype_features.view(prototype_features.shape[0], self.feature_dim)  # [len_proto, dim]
        context_prompts = context_prompts.view(context_prompts.shape[0], self.feature_dim)  # [len_context, dim]
        text_features = text_features.expand(image_features.shape[0], text_features.shape[0], self.feature_dim)  # [bs, num_text, dim]
        prototype_features = prototype_features.expand(image_features.shape[0], prototype_features.shape[0], self.feature_dim)  # [bs, len_proto, dim]
        context_prompts = context_prompts.expand(image_features.shape[0], context_prompts.shape[0], self.feature_dim)  # [bs, len_context, dim]
        features = torch.cat([image_features, text_features, prototype_features, context_prompts], dim=1)  # [bs, (1+num_text+num_proto+num_context), dim]
        features = self.sel_attn(features, features, features)
        image_features = features[:, 0, :]  # [bs, dim]
        text_features = features[:, 1:len_texts+1, :]  # [bs, num_text, dim]
        prototype_features = features[:, len_texts+1:len_texts+1+len_protos, :]  # [bs, num_proto, dim]
        context_prompts = features[:, len_texts+1+len_protos:len_texts+1+len_protos+len_context_prompts, :]  # [bs, num_context, dim]
        text_features = torch.mean(text_features, dim=0)  # [num_text, dim]
        prototype_features = torch.mean(prototype_features, dim=0)  # [num_proto, dim]
        image_features = image_features.view(image_features.shape[0], -1)
        text_features = text_features.view(text_features.shape[0], -1)
        prototype_features = prototype_features.view(prototype_features.shape[0], -1)
        return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features
    
    def forward_transformer(self, image_features, text_features, transformer=False):
        prototype_features = self.encode_prototpyes(normalize=True)
        if transformer:
            context_prompts = self.get_context_prompts()
            len_texts = text_features.shape[0]
            len_protos = prototype_features.shape[0]
            len_context_prompts = context_prompts.shape[0]
            image_features = image_features.view(image_features.shape[0], -1, self.feature_dim)  # [bs, 1, dim]
            text_features = text_features.view(text_features.shape[0], self.feature_dim)  # [total_classes, dim]
            prototype_features = prototype_features.view(prototype_features.shape[0], self.feature_dim)  # [len_pro, dim]
            context_prompts = context_prompts.view(context_prompts.shape[0], self.feature_dim)  # [len_context, dim]
            text_features = text_features.expand(image_features.shape[0], text_features.shape[0], self.feature_dim)  # [bs, total_classes, dim]
            prototype_features = prototype_features.expand(image_features.shape[0], prototype_features.shape[0], self.feature_dim)  # [bs, len_pro, dim]
            context_prompts = context_prompts.expand(image_features.shape[0], context_prompts.shape[0], self.feature_dim)  # [bs, len_context, dim]
            features = torch.cat([image_features, text_features, prototype_features, context_prompts], dim=1)
            features = self.sel_attn(features, features, features)
            image_features = features[:, 0, :]  # [bs, dim]
            text_features = features[:, 1:len_texts+1, :]  # [bs, num_text, dim]
            prototype_features = features[:, len_texts+1:len_texts+1+len_protos, :]  # [bs, num_proto, dim]
            context_prompts = features[:, len_texts+1+len_protos:len_texts+1+len_protos+len_context_prompts, :]  # [bs, num_context, dim]
            text_features = torch.mean(text_features, dim=0)  
            prototype_features = torch.mean(prototype_features, dim=0)
            image_features = image_features.view(image_features.shape[0], -1)
            text_features = text_features.view(text_features.shape[0], -1)
            prototype_features = prototype_features.view(prototype_features.shape[0], -1)
            return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features
        else:
            return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features
    
    def freeze_projection_weight_new(self):
        if len(self.projs_img) > 1:
            for i in range(len(self.projs_img) - 1):  # 除了最新的投影外都冻结
                for param in self.projs_img[i].parameters():
                    param.requires_grad = False
                for param in self.projs_text[i].parameters():
                    param.requires_grad = False
                for param in self.projs_state[i].parameters():  # 冻结旧的虫态投影
                    param.requires_grad = False
            for param in self.projs_img[-1].parameters():
                param.requires_grad = True
            for param in self.projs_text[-1].parameters():
                param.requires_grad = True
            for param in self.projs_state[-1].parameters():
                param.requires_grad = True
        for param in self.sel_attn.parameters():
            param.requires_grad = True
        # 虫态嵌入始终可训练，确保访问正确的属性 state_embedder
        for param in self.state_embedder.parameters():
            param.requires_grad = True
        # 如果需要冻结虫态嵌入，可根据需要取消下面的代码
        # for param in self.state_embedder.parameters():
        #     param.requires_grad = False

    def encode_state(self, state_ids, normalize: bool = False):
        """编码虫态ID为嵌入向量"""
        state_features = self.state_embedder.get_state_embeddings(state_ids)
        state_projections = [proj(state_features) for proj in self.projs_state]
        state_features = torch.stack(state_projections, dim=1)
        state_features = torch.sum(state_features, dim=1)
        if normalize:
            state_features = F.normalize(state_features, dim=1)
        return state_features

    def forward_tri_modal(self, image, text, state_ids):
        """三路投影融合前向传播"""
        image_features = self.encode_image(image, normalize=True)
        if isinstance(text, list):
            text_tensor = self.tokenizer(text).to(self._device)
        else:
            text_tensor = text
        text_features = self.encode_text(text_tensor, normalize=True)
        state_features = self.encode_state(state_ids, normalize=True)
        prototype_features = self.encode_prototpyes(normalize=True)
        context_prompts = self.get_context_prompts()
        len_texts = text_features.shape[0]
        len_protos = prototype_features.shape[0]
        len_context_prompts = context_prompts.shape[0]
        batch_size = image_features.shape[0]
        image_features = image_features.view(batch_size, 1, self.feature_dim)
        state_features = state_features.view(batch_size, 1, self.feature_dim)
        if text_features.shape[0] == batch_size:
            text_features = text_features.unsqueeze(1)
        else:
            text_features = text_features.view(text_features.shape[0], self.feature_dim)
            text_features = text_features.expand(batch_size, text_features.shape[0], self.feature_dim)
        prototype_features = prototype_features.view(prototype_features.shape[0], self.feature_dim)
        prototype_features = prototype_features.expand(batch_size, prototype_features.shape[0], self.feature_dim)
        context_prompts = context_prompts.view(context_prompts.shape[0], self.feature_dim)
        context_prompts = context_prompts.expand(batch_size, context_prompts.shape[0], self.feature_dim)
        features = torch.cat([
            image_features,
            text_features,
            state_features,
            prototype_features,
            context_prompts
        ], dim=1)
        features = self.sel_attn(features, features, features)
        image_output_idx = 0
        text_output_start_idx = 1
        if text_features.shape[1] == 1:
            text_output_end_idx = text_output_start_idx + 1
            state_output_idx = text_output_end_idx
        else:
            text_output_end_idx = text_output_start_idx + len_texts
            state_output_idx = text_output_end_idx
        proto_output_start_idx = state_output_idx + 1
        proto_output_end_idx = proto_output_start_idx + len_protos
        image_features = features[:, image_output_idx]
        text_features = features[:, text_output_start_idx:text_output_end_idx]
        state_features = features[:, state_output_idx]
        prototype_features = features[:, proto_output_start_idx:proto_output_end_idx]
        if text_features.shape[1] > 1:
            text_features = torch.mean(text_features, dim=1)
        if prototype_features.shape[1] > 1:
            prototype_features = torch.mean(prototype_features, dim=1)
        return image_features, text_features, state_features, prototype_features, self.convnet.logit_scale.exp()

    def evolve_state_prototypes(self):
        """使用演化图网络更新虫态原型"""
        if not hasattr(self, "img_prototypes_by_state") or not self.img_prototypes_by_state:
            return None
        evolution_results = self.state_evolution_graph.evolve_and_update(self.img_prototypes_by_state)
        evolved_protos = evolution_results['prototypes']
        evolution_embeddings = evolution_results['embeddings']
        alpha = 0.6  # 融合比例
        for class_id, state_protos in evolved_protos.items():
            for state_id, evolved_proto in state_protos.items():
                if class_id in self.img_prototypes_by_state and state_id in self.img_prototypes_by_state[class_id]:
                    original = self.img_prototypes_by_state[class_id][state_id]
                    fused = alpha * original + (1 - alpha) * evolved_proto
                    self.img_prototypes_by_state[class_id][state_id] = F.normalize(fused, dim=0)
        self._sync_class_prototypes()
        self.evolution_embeddings = evolution_embeddings
        return evolution_embeddings

    def _sync_class_prototypes(self):
        """同步更新类别原型（基于所有虫态原型的加权平均）"""
        if not hasattr(self, 'img_prototypes') or self.img_prototypes is None:
            return
        for class_id in range(len(self.img_prototypes)):
            if class_id in self.img_prototypes_by_state and self.img_prototypes_by_state[class_id]:
                protos = []
                weights = []
                for state_id, proto in self.img_prototypes_by_state[class_id].items():
                    protos.append(proto)
                    weights.append(1.5 if state_id == 4 else 1.0)
                if protos:
                    weights = torch.tensor(weights, device=protos[0].device)
                    weights = weights / weights.sum()
                    weighted_sum = torch.zeros_like(protos[0])
                    for i, proto in enumerate(protos):
                        weighted_sum += weights[i] * proto
                    self.img_prototypes[class_id] = F.normalize(weighted_sum, dim=0)


