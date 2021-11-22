import torch
import torch.nn as nn
import numpy as np

from networks.resnet import resnet101


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class MLP(nn.Module):
    def __init__(self, channels=[4096, 1024, 2048]):
        super(MLP, self).__init__()
        layer_num = len(channels) - 1
        
        linear_list = []
        for i in range(layer_num):
            linear_list.append(nn.Linear(channels[i], channels[i+1]))
            if i != layer_num - 1:
                linear_list.append(nn.Dropout())
        
        self.layers = nn.Sequential(*linear_list)
    
    def forward(self, input):
        out = self.layers(input)
        return out

class APINet(nn.Module):

    def __init__(self, num_classes=200):
        super(APINet, self).__init__()
        conv = resnet101(pretrained=True, n_classes=num_classes)
        conv.fc = nn.Identity()
        conv.avgpool = nn.AvgPool2d(kernel_size=14, stride=1)
        self.conv = conv
        
        self.mlp = MLP(channels=[4096, 512, 2048])
        self.fc = nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=0.5)
        

    def forward(self, imgs, targets=None, flag=False):
        '''
        return:
            - out_prob: 分类概率 8N x 200
            - inter_labels: 选出的不同类的标签 N
        '''
        if flag=='val':
            feat = self.conv(imgs)
            out = self.fc(feat)
            return out
        
        feats = self.conv(imgs)                         # feats:    N  x 2048
        
        # 配对，组成 pair
        intra_feats, inter_feats, inter_labels = self.match_pair(feats, targets)
        x_feat = torch.cat((feats, feats), dim=0)               # 2N x 2048
        y_feat = torch.cat((intra_feats, inter_feats), dim=0)   # 2N x 2048
        
        # Mutual Vector
        m_feat = self.mlp(torch.cat([x_feat, y_feat], axis=1))
        
        # Gate Vector Generation
        gate_x = torch.mul(m_feat, x_feat)
        gate_y = torch.mul(m_feat, y_feat)
        gate_x = self.sigmoid(gate_x)
        gate_y = self.sigmoid(gate_y)
        
        # Pairwise Interaction
        x_self  = x_feat + torch.mul(x_feat, gate_x)            # 2N x 2048
        x_other = x_feat + torch.mul(x_feat, gate_y)
        y_self  = y_feat + torch.mul(y_feat, gate_y)
        y_other = y_feat + torch.mul(y_feat, gate_x)
        
        # out_feats = torch.cat((x_self, x_other, y_self, y_other))   # 8N x 2048
        # out_prob = self.fc(out_feats)                               # 8N x 200

        logit1_self = self.sigmoid(self.fc(self.drop(x_self)))
        logit1_other = self.sigmoid(self.fc(self.drop(x_other)))
        logit2_self = self.sigmoid(self.fc(self.drop(y_self)))
        logit2_other = self.sigmoid(self.fc(self.drop(y_other)))

        labels1 = torch.cat([targets, targets], dim=0)
        labels2 = torch.cat([targets, inter_labels], dim=0)
        
        return logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2
        return out_prob, inter_labels
    
    def match_pair(self, feats, labels):
        '''
            计算样本特征的欧式距离，对每个样本找到与他距离最近的 同类/不同类 的样本
            - N   N
            - N   n
            feats:  N x 2048
            labels: N
            
            return: 
            - intra_feats: 选出的同类特征   N x 2048
            - inter_feats: 选出的不同类特征 N x 2048
            - inters: 不同类特征的标签      N
        '''
        N = labels.shape[0]
        labels = labels.detach().cpu().numpy()
        
        # 计算欧氏距离矩阵 N x N
        matrix = np.zeros((N, N))
        for i, f in enumerate(feats):
            for j, f_other in enumerate(feats):
                matrix[i, j] = torch.sqrt(torch.sum(torch.square(f - f_other))).item()
        
        # intras, inters = [], []
        # 先同类
        m_intra = np.array(matrix)
        # 不同类的距离先置为 inf
        for i in range(N):
            mask = labels==labels[i]
            m_intra[i, mask==False] = np.inf
        indices_intra = np.argsort(m_intra)
        intras = indices_intra[:,1]
        # for i in range(N):
        #     i_intra = indices[i, mask][1]   # 第 0 个是自身
        #     intras.append(i_intra.item())

        # 再不同类
        m_inter = np.array(matrix)
        # 同类的距离先置为 inf
        for i in range(N):
            mask = labels==labels[i]
            m_inter[i, mask==True] = np.inf
        indices_inter = np.argsort(m_inter)
        inters = indices_inter[:,0]
        # for i in range(N):
        #     i_inter = indices[i, mask][1]   # 第 0 个是自身
        #     intras.append(i_inter.item())


        # # 排序后获得距离最小的 intra 和 inter 下标
        # intras, inters = [], []
        # dist, indices = torch.sort(matrix, dim=-1)
        # for i in range(N):
        #     # 同类
        #     mask = labels==labels[i]
        #     matrix[i, mask==False] = np.inf
        #     i_intra = indices[i, mask][1]   # 第 0 个是自身
        #     intras.append(i_intra.item())
        #     # 不同类
        #     mask = ~mask
        #     i_inter = indices[i, mask][0]
        #     inters.append(i_inter.item())
        
        # 组成 特征
        intra_feats = feats[intras] # N x 2048
        inter_feats = feats[inters]
        
        return intra_feats, inter_feats, torch.tensor(labels[inters]).to(device=device)
        

        