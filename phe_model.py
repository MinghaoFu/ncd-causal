import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from vision_transformer import vit_base, Mlp, ViTDecoder

def dag_constraint(matrix):
    # DAG constraint from DAG-GNN
    # Code obtained from https://github.com/fishmoon1234/DAG-GNN/blob/2f8688908d9ad3d85879feb97ff59ff1f73ed151/src/train.py#L261C1-L265C15
    num_var = len(matrix)
    x = torch.eye(num_var) + torch.div(matrix * matrix, num_var)
    return torch.trace(x) - num_var

class PrototypeMask(nn.Module):
    def __init__(self, p=0.5):
        super(PrototypeMask, self).__init__()
        self.p = p
        self.mask = None

    def forward(self, X):
        if self.training:
            self.mask = torch.bernoulli(torch.full_like(X, 1 - self.p))
            return X * self.mask
        else:
            return X * (1 - self.p)

class GenHash(nn.Module):   
    def __init__(self, hash_code_length, y_embed_dim=128, y_hid_dim=128, e_hid_dim=32, map_hid_dim=256, max_labels=200):
        super(GenHash, self).__init__()
        # y head
        self.y_embed_layer = nn.Embedding(num_embeddings=max_labels, embedding_dim=y_embed_dim)
        self.y_layer = Mlp(y_embed_dim, (y_embed_dim + y_hid_dim) // 2, y_hid_dim)    
        self.e_layer = Mlp(1, (1 + e_hid_dim) // 2, e_hid_dim)
        self.hash = nn.ModuleList(
            [Mlp(y_hid_dim + e_hid_dim, map_hid_dim, 1) for _ in range(hash_code_length)]        
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, y, e):
        y_embed = self.y_embed_layer(y).squeeze(1)  
        y_h = self.y_layer(y_embed)
        h_list = [] 
        for i in range(len(self.hash)): 
            e_h = self.e_layer(e[:, i:i+1])
            h_list.append(self.hash[i](torch.cat([y_h, e_h], dim=-1)))
        #h = nn.Tanh()(torch.cat(h_list, dim=-1) * 3)
        h = torch.cat(h_list, dim=-1)   
        h = torch.nn.Tanh()(h*3)
        return h, y_embed

class HASHHead(nn.Module):
    def __init__(self, in_dim, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256, code_dim=12):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            layers.append(nn.BatchNorm1d(bottleneck_dim)) ## new
            layers.append(nn.GELU()) ## new
            self.mlp = nn.Sequential(*layers)
        
        self.hash = nn.Linear(bottleneck_dim, code_dim, bias=False)
        self.bn_h = nn.BatchNorm1d(code_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.mlp(x)
        h = self.hash(z)
        return h

# MLP Decoder
class MLPDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.fc_out(x)
        return x
    
class PPNet_Normal(nn.Module):
    def __init__(self, args, features, img_size, prototype_shape,
                 num_classes,
                 use_global=False,
                 global_proto_per_class=10,
                 init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck',
                 mask_theta=0.1,
                 hash_code_length=12,
                 zs_dim=256):

        super(PPNet_Normal, self).__init__()
        self.args = args
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes

        self.use_global = use_global

        self.global_proto_per_class = global_proto_per_class
        self.epsilon = 1e-4

        self.num_prototypes_global = self.num_classes * self.global_proto_per_class

        self.prototype_shape_global = [self.num_prototypes_global] + [self.prototype_shape[1]]

        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)

        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)
        self.prototype_class_identity_global = torch.zeros(self.num_prototypes_global,
                                                    self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        num_prototypes_per_class_global = self.num_prototypes_global // self.num_classes
        for j in range(self.num_prototypes_global):
            self.prototype_class_identity_global[j, j // num_prototypes_per_class_global] = 1

        self.features = features

        first_add_on_layer_in_channels = [i for i in features.modules() if isinstance(i, nn.Linear)][-1].out_features

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Linear(first_add_on_layer_in_channels, self.args.zs_dim),
                nn.GELU()
            )
        #self.recover = nn.Linear(self.prototype_shape[1], self.args.zs_dim)
        if self.use_global:
            self.prototype_vectors_global = nn.Parameter(torch.rand(self.prototype_shape_global),
                                              requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias
        self.last_layer_global = nn.Linear(self.num_prototypes_global, self.num_classes,
                                    bias=False) # do not use bias
        self.last_layer.weight.requires_grad = False
        self.last_layer_global.weight.requires_grad = False

        self.all_attn_mask = None
        self.teacher_model = None

        # reconstruction cls tokens
        self.cls_tokens_decoder = MLPDecoder(input_dim=self.args.zs_dim + self.args.zc_dim, output_dim=768)
        self.decoder = ViTDecoder(embed_dim=768)
        self.flow = Mlp(hash_code_length, self.args.zs_dim * 2, self.args.zs_dim)
        self.logvar = nn.Parameter(torch.zeros(self.args.zs_dim), requires_grad=True)  
        # for learning zc
        if self.args.zc_dim > 0:    
            self.zc_encoder = Mlp(768, (self.prototype_shape[1] + self.args.zc_dim) // 2, self.args.zc_dim)     

        # hash to z flow
        self.hash_flows = nn.ModuleList([Mlp(2, 4, 1) for _ in range(hash_code_length)]) 

        # mask
        self.eps_d_mask_logits = nn.Parameter(torch.zeros(hash_code_length), requires_grad=True)  
        # recon d from y and epsilon
        self.gen_hash = GenHash(hash_code_length, max_labels=args.unlabeled_nums)

        self.scale = self.prototype_shape[1] ** -0.5

        if init_weights:
            self._initialize_weights()

        self.hash_head = HASHHead(in_dim=self.args.zs_dim, code_dim=hash_code_length)

        self.prototypemask = PrototypeMask(p=mask_theta)

        for param in self.features.parameters():
            param.requires_grad = False

        for param in self.features.blocks[-1].parameters():
            param.requires_grad = True


    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution_single(self, x, prototype_vectors):
        temp_ones = torch.ones(prototype_vectors.shape).cuda()

        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=temp_ones)

        p2 = prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances


    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def get_activations(self, tokens, prototype_vectors):
        # distance 
        tokens_expanded = F.normalize(tokens, dim=-1)
        prototype_vectors_expanded = F.normalize(prototype_vectors, dim=-1)
        # print("tokens_expanded.shape", tokens_expanded.shape)
        # print("prototype_vectors_expanded.shape", prototype_vectors_expanded.shape)
        tokens_expanded = tokens_expanded.unsqueeze(1).expand(-1, prototype_vectors.size(0), -1)
        prototype_vectors_expanded = prototype_vectors_expanded.unsqueeze(0).expand(tokens.size(0), -1, -1)
        diff = tokens_expanded - prototype_vectors_expanded
        diff_squared = diff ** 2
        euclidean_distances = diff_squared.sum(dim=2).sqrt()
        # print(euclidean_distances)
        # print("euclidean_distances.shape", euclidean_distances.shape)
        ## distance 
        activations = self.distance_2_similarity(euclidean_distances)   # (B, 2000, 1, 1)
        total_proto_act = activations

        if self.use_global:
            return activations, (euclidean_distances, total_proto_act)
        return activations

    def forward(self, x, syn_cls_tokens=None):

        if not self.training:

            _cls_tokens, patch_tokens, frz_cls_tokens, frz_patch_tokens = self.features.forward_all(x)
            
            cls_tokens = self.add_on_layers(_cls_tokens)
            # cls_tokens = nn.ReLU()(cls_tokens)  
            if self.args.zc_dim > 0:
                zc = self.zc_encoder(_cls_tokens)
            else:
                zc = torch.empty(x.shape[0], 0).to(x.device)
                
            hash_feat = self.hash_head(cls_tokens)
            
            global_activations, _ = self.get_activations(cls_tokens, self.prototype_vectors_global)

            # logits_global = self.last_layer_global(global_activations)

            return hash_feat, frz_cls_tokens, frz_patch_tokens, cls_tokens, patch_tokens, zc
 
        x = self.features.forward_frz_blocks(x)
        if syn_cls_tokens is not None:
            x[:, 0] = syn_cls_tokens
        frz_cls_tokens, frz_patch_tokens = x[:, 0], x[:, 1:]
        _cls_tokens, patch_tokens = self.features.forward_last_block(x)
        
        cls_tokens = self.add_on_layers(_cls_tokens)
        # cls_tokens = nn.ReLU()(cls_tokens)  
        if self.args.zc_dim > 0:
            zc = self.zc_encoder(_cls_tokens) 
        else:
            zc = torch.empty(x.shape[0], 0).to(x.device)

        hash_feat = self.hash_head(cls_tokens)

        global_activations, _ = self.get_activations(cls_tokens, self.prototype_vectors_global)

        global_activations = self.prototypemask(global_activations)

        logits_global = self.last_layer_global(global_activations)
        
        return logits_global, self.features, hash_feat, frz_cls_tokens, frz_patch_tokens, cls_tokens, patch_tokens, zc


    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

        if hasattr(self, 'last_layer_global'):
            positive_one_weights_locations = torch.t(self.prototype_class_identity_global)
            negative_one_weights_locations = 1 - positive_one_weights_locations

            self.last_layer_global.weight.data.copy_(
                correct_class_connection * positive_one_weights_locations
                + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                trunc_normal_(m.weight, std=.02)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


def construct_PPNet_dino(args, img_size=224,
                    prototype_shape=(2000, 512, 1, 1), num_classes=200,
                    use_global=False,
                    global_proto_per_class=10,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck',
                    mask_theta=0.1,
                    pretrain_path='',
                    hash_code_length=12):
    features = vit_base()
    features.load_state_dict(torch.load(pretrain_path))

    return PPNet_Normal(args, features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 num_classes=num_classes,
                 use_global=use_global,
                 global_proto_per_class=global_proto_per_class,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type,
                 mask_theta=mask_theta,
                 hash_code_length=hash_code_length) 
