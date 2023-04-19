import torch.nn.functional
import torchvision
from torch.nn.init import normal, constant
from model.stgcn import *
from .ops.basic_ops import ConsensusModule


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / 2. / logvar.exp()).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class Basic_block(nn.Module):
    def __init__(self, in_feature, hidden, out_feature):
        super().__init__()
        self.fcn1 = nn.Linear(in_feature, out_feature)
        self.Leakyrelu1 = nn.LeakyReLU(0.2)
        self.fcn2 = nn.Linear(out_feature, hidden)
        self.Leakyrelu2 = nn.LeakyReLU(0.2)
        self.fcn3 = nn.Linear(hidden, out_feature)
        self.Leakyrelu3 = nn.LeakyReLU(0.2)
        self.drop_out = nn.Dropout(0.2)

    def forward(self, x):
        x1 = self.drop_out(self.Leakyrelu1(self.fcn1(x)))
        x = self.drop_out(self.Leakyrelu2(self.fcn2(x1)))
        x = self.drop_out(self.Leakyrelu3(self.fcn3(x))) + x1


class CPC(nn.Module):
    """
        Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.

        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """

    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)
        if n_layers == 1:
            self.net = nn.Linear(
                in_features=y_size,
                out_features=x_size
            )
        else:
            net = nn.ModuleList()
            for i in range(n_layers):
                if i == 0:
                    net.append(nn.Linear(self.y_size, self.x_size))
                    net.append(self.activation())
                else:
                    net.append(nn.Linear(self.x_size, self.x_size))
            self.net = nn.Sequential(*net)

    def forward(self, x, y):
        """Calulate the score
        """
        # import ipdb;ipdb.set_trace()
        x_pred = self.net(y)  # bs, emb_size

        # normalize to unit sphere
        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True)

        pos = torch.sum(x * x_pred, dim=-1)  # bs
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)  # bs
        nce = -(pos - neg).mean()
        return nce


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality, modality_1,
                 base_model='resnet18', new_length=None, new_length_1=None,
                 embed=False, consensus_type='avg', before_softmax=True,
                 dropout=0.8, crop_num=1, partial_bn=True, context=False,  args=None):
        super(TSN, self).__init__()
        # 第一种模态
        self.modality = modality
        # 第二种模态
        self.modality_1 = modality_1
        # 视频分段的数量
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.args = args
        self.consensus_type = consensus_type
        self.embed_dim = self.args.embed_dim
        self.embed = embed
        self.embed_1 = False
        if modality == "RGB":
            self.new_length = 1
        else:
            self.new_length = new_length
        if self.modality_1 == 'Flow':
            self.new_length_1 = 5
        else:
            self.new_length_1 = new_length_1

        """st_gcn 模块 输入通道，输出情感分类以及回归 openpose18个肢体动作的点构成 """
        self.in_channels = 3
        self.num_classes = 26
        self.num_dimensions = 3
        self.layout = 'openpose'
        self.strategy = 'spatial'
        self.max_hop = 1
        self.dilation = 1
        self.edge_importance_weighting = True
        self.model = Model(in_channels=self.in_channels, num_class=self.num_classes, num_dim=self.num_dimensions,
                           layout=self.layout, strategy=self.strategy, max_hop=self.max_hop, dilation=self.dilation,
                           edge_importance_weighting=self.edge_importance_weighting)
        pretrained_dict = torch.load('/home/a/Documents/MBGE/stgcn/models/kinetics-st_gcn.pt')
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        """ 这一部分构建时空图卷积的网络，并且网络的参数用预备训练的网络进行初始化"""

        """构建第一个模态的模型，采用resnet构建模型"""
        self.name_base = base_model
        self.context = context
        if self.modality == 'RGB':
            print(("""
            Initializing TSN with base model: {}.
            TSN Configurations:
                input_modality:     {}
                num_segments:       {}
                new_length:         {}
                consensus_module:   {}
                dropout_ratio:      {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))
            self._prepare_base_model(base_model)
            if context:
                self._prepare_context_model()
            self.feature_dim = self._prepare_tsn(num_class)
        """可以选择是否采用context"""

        """构建第二个模态的模型，采用resnet构建模型"""
        self.context_1 = False
        if self.modality_1 == 'Flow':
            print(("""
            Initializing TSN with base model: {}.
            TSN Configurations:
                input_modality:     {}
                num_segments:       {}
                new_length:         {}
                consensus_module:   {}
                dropout_ratio:      {}
            """.format(base_model, self.modality_1, self.num_segments, self.new_length_1, consensus_type, self.dropout)))
            self._prepare_base_model_1(base_model)
            self.feature_dim_1 = self._prepare_tsn_1(num_class)
            self.base_model_1 = self._construct_flow_model(self.base_model_1)
            print("Converting the ImageNet model to a flow init model")
            print("Done. Flow model ready...")
            if self.context_1:
                self._prepare_context_model_1()
                self.context_model_1 = self._construct_flow_model(self.context_model_1)
                print("Converting the context model to a flow init model")
                print("Done. Flow model ready...")
        """可以选择是否采用context"""

        """是否采用batchnorm"""
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)
        """是否采用batchnorm"""

        """最终结果取平均模型"""
        self.consensus = ConsensusModule(consensus_type)
        self.consensus_cont = ConsensusModule(consensus_type)
        """最终结果取平均模型"""

        """对embed representation 取平均"""
        if self.embed:
            self.consensus_embed = ConsensusModule(consensus_type)
        """对embed representation 取平均"""

        """这里需要设置最终的st_gcn_dim，以及对应映射维度"""
        self.st_gcn_dim = args.st_gcn_dim
        if args.opn == 'cat':
            if self.st_gcn_dim == 256:
                self.input_dim = self.feature_dim + self.feature_dim_1 + self.st_gcn_dim
            else:
                self.st_gcn_linear = nn.Linear(256, self.st_gcn_dim)
                self.input_dim = self.feature_dim + self.feature_dim_1 + self.st_gcn_dim
        """如果不是256那么就需要对其进行映射"""

        """这里需要设置最终的st_gcn_dim，以及对应映射维度"""
        if args.opn == 'mul':
            if self.st_gcn_dim == 256:
                self.input_dim = self.feature_dim + self.st_gcn_dim
            else:
                self.st_gcn_linear = nn.Linear(256, self.st_gcn_dim)
                self.input_dim = self.feature_dim + self.st_gcn_dim
        """如果不是256那么就需要对其进行映射"""

        """预测全连接网络"""
        # self.linear_class = nn.Sequential(nn.Linear(self.input_dim, 1024), nn.Linear(1024, 26))
        # self.linear_continue = nn.Sequential(nn.Linear(self.input_dim, 1024), nn.Linear(1024, 3))
        """预测全连接网络"""
        #######
        self.linear_class = nn.Sequential(nn.Linear((self.feature_dim+self.feature_dim_1)*self.num_segments+self.st_gcn_dim, 1024), nn.Linear(1024, 26))
        self.linear_continue = nn.Sequential(nn.Linear((self.feature_dim+self.feature_dim_1)*self.num_segments+self.st_gcn_dim, 1024), nn.Linear(1024, 3))
        #######

        """用于计算聚类损失的网络"""
        self.center_dim = self.args.center_dim
        self.consensus_for_loss = ConsensusModule(consensus_type)
        # self.project = nn.Sequential(nn.Linear(self.input_dim, 1024), nn.Linear(1024, self.center_dim))
        #######
        self.project = nn.Sequential(nn.Linear((self.feature_dim+self.feature_dim_1)*self.num_segments+self.st_gcn_dim, 1024), nn.Linear(1024, self.center_dim))
        #######
        """用于计算聚类损失的网络"""

        """对比学习，对比预测编码"""
        self.CPC_flow_stgcn = CPC(self.st_gcn_dim, self.feature_dim_1, n_layers=3)
        self.CPC_RGB_stgcn = CPC(self.st_gcn_dim, self.feature_dim, n_layers=3)
        """对比学习，对比预测编码"""
    def forward(self, input, input_1, input2):
        """resnet的通道数量对于不同模态是不一样的"""
        if self.modality == "RGB":
            sample_len = 3
        if self.modality_1 == 'Flow':
            sample_len_1 = 10

        """RGB模态前向传播"""
        if self.context:
            inp = input.view((-1, sample_len) + input.size()[-2:])
            body_indices = list(range(0, inp.size(0), 2))
            context_indices = list(range(1, inp.size(0), 2))
            body = inp[body_indices]
            context = inp[context_indices]
        else:
            body = input.view((-1, sample_len) + input.size()[-2:])
        base_out = self.base_model(body).squeeze(-1).squeeze(-1)
        if self.context:
            context_out = self.context_model(context).squeeze(-1).squeeze(-1)
        """RGB模态前向传播"""

        """Flow模态前向传播"""
        if self.context_1:
            inp_1 = input_1.view((-1, sample_len_1) + input_1.size()[-2:])
            body_indices_1 = list(range(0, inp_1.size(0), 2))
            context_indices_1 = list(range(1, inp_1.size(0), 2))
            body_1 = inp[body_indices_1]
            context_1 = inp[context_indices_1]
        else:
            body_1 = input_1.view((-1, sample_len_1) + input_1.size()[-2:])
        base_out_1 = self.base_model_1(body_1).squeeze(-1).squeeze(-1)
        if self.context_1:
            context_out_1 = self.context_model_1(context_1).squeeze(-1).squeeze(-1)
        """Flow模态前向传播"""

        outputs = {}

        """对embed进行处理，如果有的话"""
        if self.embed:
            embed_segm = self.embed_fc(base_out)
            embed = embed_segm.view((-1, self.num_segments) + embed_segm.size()[1:])
            embed = self.consensus_embed(embed).squeeze(1)
            outputs['embed'] = embed
        if self.embed_1:
            embed_segm_1 = self.embed_fc_1(base_out_1)
            embed_1 = embed_segm_1.view((-1, self.num_segments) + embed_segm_1.size()[1:])
            embed_1 = self.consensus_embed(embed_1).squeeze(1)
            outputs['embed_1'] = embed_1
        """对embed进行处理，如果有的话"""

        """融合，一般不需要"""
        if self.context:
            base_out = torch.mul(base_out, context_out)
        if self.context_1:
            base_out_1 = torch.mul(base_out_1, context_out_1)
        """融合，一般不需要"""

        """对双流网络进行融合"""
        if self.args.opn == 'cat':
            resnet_output = torch.cat([base_out_1, base_out], dim=1)
        else:
            resnet_output = torch.mul(base_out_1, base_out)
        """对双流网络进行融合"""

        """stgcn前向传播"""
        out_stgcn = self.model(input2)
        if self.st_gcn_dim != 256:
            out_stgcn = self.st_gcn_linear(out_stgcn)
        if self.num_segments != 1:
            out_stgcn_1 = out_stgcn.unsqueeze(1).repeat(1, self.num_segments, 1)
            out_stgcn_2 = out_stgcn_1.view(-1, out_stgcn_1.shape[-1])
        else:
            out_stgcn_2 = out_stgcn
        ##########
        resnet_output_2 = torch.cat([resnet_output.view(-1, resnet_output.shape[-1] * self.num_segments), out_stgcn], dim=1)
        ##########
        # resnet_output = torch.cat([resnet_output, out_stgcn_2], dim=1)
        """stgcn前向传播"""

        """进行前向预测，然后对一个视频的多个片段上的结果取平均"""
        ##########
        results_class = self.linear_class(resnet_output_2)
        results_continue = self.linear_continue(resnet_output_2)
        outputs['categorical'] = results_class
        outputs['continuous'] = results_continue
        ##########
        # results_class = self.linear_class(resnet_output)
        # results_continue = self.linear_continue(resnet_output)
        # base_out_cat = results_class.view((-1, self.num_segments) + results_class.size()[1:])
        # base_out_cont = results_continue.view((-1, self.num_segments) + results_continue.size()[1:])
        # output = self.consensus(base_out_cat)
        # outputs['categorical'] = output.squeeze(1)
        # output_cont = self.consensus_cont(base_out_cont)
        # outputs['continuous'] = output_cont.squeeze(1)

        """进行前向预测，然后对一个视频的多个片段上的结果取平均"""
        ##########
        feature = self.project(resnet_output_2)
        ##########
        """得到特征用于聚类"""
        # feature = self.project(resnet_output)
        # feature = self.consensus_for_loss(feature.view((-1, self.num_segments) + feature.size()[1:])).squeeze(1)
        """得到特征用于聚类"""

        """对比预测编码损失，最大化相关信息"""
        NCE_loss = 0
        NCE_loss += self.CPC_RGB_stgcn(out_stgcn_2, base_out)
        NCE_loss += self.CPC_flow_stgcn(out_stgcn_2, base_out_1)
        """对比预测编码损失，最大化相关信息"""
        return outputs, feature, NCE_loss

    """club方法需要传入相应的模型，然后传入需要进行计算互信息的表示，以及需要解耦的num_factor"""
    def loss_dependence_club_b(self, model, representation, num_factors):
        mi_loss = 0.
        cnt = 0
        for i in range(num_factors):
            for j in range(i + 1, num_factors):
                mi_loss += model[cnt](representation[:, i * self.dim_2: (i + 1) * self.dim_2],
                                      representation[:, j * self.dim_2: (j + 1) * self.dim_2])
                cnt += 1
        return mi_loss
    """club方法需要传入相应的模型，然后传入需要进行计算互信息的表示，以及需要解耦的num_factor"""

    """club方法需要传入相应的模型，然后传入需要进行计算互信息的表示，以及需要解耦的num_factor"""
    def lld_bst(self, model, representation, num_factors):
        cnt = 0
        lld_loss = 0
        for i in range(num_factors):
            for j in range(i + 1, num_factors):
                lld_loss += model[cnt].learning_loss(representation[:, i * self.dim_2: (i + 1) * self.dim_2],
                                                     representation[:, j * self.dim_2: (j + 1) * self.dim_2])
                cnt += 1
        return lld_loss
    """club方法需要传入相应的模型，然后传入需要进行计算互信息的表示，以及需要解耦的num_factor"""

    """初始化RGB模态的模型"""
    def _prepare_tsn(self, num_class):
        std = 0.001

        if isinstance(self.base_model, torch.nn.modules.container.Sequential):
            feature_dim = 2048
        else:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
            if self.dropout == 0:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
                self.new_fc = None
            else:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))

        num_feats = 2048

        if self.embed:
            self.embed_fc = nn.Sequential(nn.Linear(num_feats, 512),
                                          nn.Linear(512, 300))
            for m in self.embed_fc:
                if isinstance(m, nn.Linear):
                    normal(m.weight, 0, std)
                    constant(m.bias, 0)

        return num_feats
    """初始化RGB模态的模型"""

    """初始化flow模态的模型"""
    def _prepare_tsn_1(self, num_class):
        std = 0.001

        if isinstance(self.base_model_1, torch.nn.modules.container.Sequential):
            feature_dim = 2048
        else:
            feature_dim = getattr(self.base_model_1, self.base_model_1.last_layer_name).in_features
            if self.dropout == 0:
                setattr(self.base_model_1, self.base_model_1.last_layer_name, nn.Linear(feature_dim, num_class))
                self.new_fc = None
            else:
                setattr(self.base_model_1, self.base_model_1.last_layer_name, nn.Dropout(p=self.dropout))

        num_feats = 2048

        if self.embed_1:
            self.embed_fc_1 = nn.Sequential(nn.Linear(num_feats, 512),
                                            nn.Linear(512, 300))
            for m in self.embed_fc_1:
                if isinstance(m, nn.Linear):
                    normal(m.weight, 0, std)
                    constant(m.bias, 0)

        return num_feats
    """初始化flow模态的模型"""

    """是否需要context模态的模型"""
    def _prepare_context_model(self):
        self.context_model = getattr(torchvision.models, "resnet50")(True)
        modules = list(self.context_model.children())[:-1]  # delete the last fc layer.
        self.context_model = nn.Sequential(*modules)

    def _prepare_context_model_1(self):
        self.context_model_1 = getattr(torchvision.models, "resnet50")(True)
        modules = list(self.context_model_1.children())[:-1]  # delete the last fc layer.
        self.context_model_1 = nn.Sequential(*modules)
    """是否需要context模态的模型"""

    """对RGB模态的模型进行修改"""
    def _prepare_base_model(self, base_model):
        import torchvision.models

        if 'resnet' in base_model or 'vgg' in base_model or 'resnext' in base_model or 'densenet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))
    """对RGB模态的模型进行修改"""

    """对flow模态的模型进行修改"""
    def _prepare_base_model_1(self, base_model):
        import torchvision.models

        if 'resnet' in base_model or 'vgg' in base_model or 'resnext' in base_model or 'densenet' in base_model:
            self.base_model_1 = getattr(torchvision.models, base_model)(True)
            self.base_model_1.last_layer_name = 'fc'
            self.input_size_1 = 224
            self.input_mean_1 = [0.485, 0.456, 0.406]
            self.input_std_1 = [0.229, 0.224, 0.225]
            if self.modality_1 == 'Flow':
                self.input_mean_1 = [0.5]
                self.input_std_1 = [np.mean(self.input_std)]

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))
    """对flow模态的模型进行修改"""

    """对模型的层的BN进行规定"""
    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            count = 0
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

            print("Freezing BatchNorm2D except the first one.")
            count = 0
            for m in self.base_model_1.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

            if self.context:
                count = 0
                print("Freezing BatchNorm2D except the first one.")
                for m in self.context_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False

            if self.context_1:
                count = 0
                print("Freezing BatchNorm2D except the first one.")
                for m in self.context_model_1.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
    """对模型的层的BN进行规定"""

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        return self.parameters()

    """对flow模型的第一层进行修改"""
    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length_1,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv = nn.Conv2d(2 * self.new_length_1, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data
            # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]
        # remove .weight suffix to get the layer name
        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model
    """对flow模型的第一层进行修改"""