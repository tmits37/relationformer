import torch
import torch.nn as nn
from copy import deepcopy
from utils import scores_to_permutations, permutations_to_polygons


def MultiLayerPerceptron(channels: list, batch_norm=True): # [128,128,64]
    n_layers = len(channels) # 채널 수: 3

    layers = []
    for i in range(1, n_layers): # 1 ~ 채널 수 - 1: 1(128->128), 2(128->64)
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True)) # (0,1),(1,2),...,(n-2,n-1)

        if i < (n_layers - 1): # 마지막 레이러 빼고 배치놈+렐루 추가하기
            if batch_norm:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())

    return nn.Sequential(*layers) # layers로 이루어진 모듈


class Attention(nn.Module):

    def __init__(self, n_heads: int, d_model: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.dim = d_model // n_heads
        self.n_heads = n_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        b = query.size(0)
        query, key, value = [l(x).view(b, self.dim, self.n_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]

        b, d, h, n = query.shape
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / d**.5
        attn = torch.einsum('bhnm,bdhm->bdhn', torch.nn.functional.softmax(scores, dim=-1), value)

        return self.merge(attn.contiguous().view(b, self.dim*self.n_heads, -1))


class AttentionalPropagation(nn.Module):

    def __init__(self, feature_dim: int, n_heads: int): # 64, 4
        super().__init__()
        self.attn = Attention(n_heads, feature_dim)
        self.mlp = MultiLayerPerceptron([feature_dim*2, feature_dim*2, feature_dim]) # 128, 128, 64
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x):
        message = self.attn(x, x, x) # query, key, value, message is attention result
        return self.mlp(torch.cat([x, message], dim=1)) # the vertex descriptor is updated. x(l+1) = MLP(cat(x(l),a(l)))


class AttentionalGNN(nn.Module):

    def __init__(self, feature_dim: int, num_layers: int): # 64, 4
        super().__init__()
        self.conv_init = nn.Sequential(
            nn.Conv1d(feature_dim + 2, feature_dim, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1,stride=1,padding=0,bias=True)
        )
        # 1~L까지 어텐션 GNN 레이어 모듈 리스트에 담기
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4) # 64, 4
            for _ in range(num_layers)]) # 64+64 -> 64 -> 64+64-> 64... 4번 반복

        self.conv_desc = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )

        self.conv_offset = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, 2, kernel_size=1,stride=1,padding=0,bias=True),
            nn.Hardtanh()
        )

    def forward(self, feat, graph):
        graph = graph.permute(0,2,1)
        feat = torch.cat((feat, graph), dim=1)
        feat = self.conv_init(feat)

        for layer in self.layers:
            feat = feat + layer(feat) # feat는 64차원
        # 마지막 레이어는 relu 안 하고 여기서 desc랑 offset으로 decomposed
        desc = self.conv_desc(feat) # matching descriptor, 노드간 타당한 연결을 생성하는데 사용된다
        offset = self.conv_offset(feat).permute(0,2,1) # positional offset, 노드 위치와 합쳐진다. p = p+a*t, -1~1 값을 가진다
        return desc, offset


class ScoreNet(nn.Module): # 가능한 모든 노드 페어를 계산하고 총 점수를 최대화한다

    def __init__(self, in_ch):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        n_points = x.shape[-1]

        x = x.unsqueeze(-1)
        x = x.repeat(1,1,1,n_points)
        t = torch.transpose(x, 2, 3)
        x = torch.cat((x, t), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        return x[:,0]


class OptimalMatching(nn.Module):

    def __init__(self):
        super(OptimalMatching, self).__init__()
        
        # Default configuration settings
        self.descriptor_dim = 64
        self.sinkhorn_iterations = 100
        self.attention_layers = 4
        self.correction_radius = 0.05

        # Modules
        self.scorenet1 = ScoreNet(self.descriptor_dim * 2) # 시계방향 permutation matrix
        self.scorenet2 = ScoreNet(self.descriptor_dim * 2) # 반시계방향 permutation matrix
        self.gnn = AttentionalGNN(self.descriptor_dim, self.attention_layers)
            

    def normalize_coordinates(self, graph, ws, input):
        if input == 'global':
            graph = (graph * 2 / ws - 1)
        elif input == 'normalized':
            graph = ((graph + 1) * ws / 2)
            graph = torch.round(graph).long()
            graph[graph < 0] = 0
            graph[graph >= ws] = ws - 1
        return graph


    def predict(self, feature_map, graph):
        B, _, H, W = feature_map.shape
        B, N, _ = graph.shape # B N 2

        for b in range(B): # 배치 사이즈만큼 반복하여 전체 디스크립터(sel_desc) 만들기
            b_desc = feature_map[b]
            b_graph = graph[b]

            # Extract descriptors 디스크립터 뽑기
            b_desc = b_desc[:, b_graph[:,0], b_graph[:,1]]

            # Concatenate descriptors in batches
            if b == 0:
                sel_desc = b_desc.unsqueeze(0)
            else:
                sel_desc = torch.cat((sel_desc, b_desc.unsqueeze(0)), dim=0) # 최종적으로 B, D, N (B, 64,256)

        # Multi-layer Transformer network.
        norm_graph = self.normalize_coordinates(graph, W, input="global") #out: normalized coordinate system [-1, 1]
        sel_desc, offset = self.gnn(sel_desc, norm_graph) # gnn 실행하여 매칭 디스크립터와 좌표 오프셋 뽑기

        # Correct points coordinates
        norm_graph = norm_graph + offset * self.correction_radius # 좌표 refine
        graph = self.normalize_coordinates(norm_graph, W, input="normalized") # out: global coordinate system [0, W]

        # Compute scores 옵티멀 커넥션 네트워크
        scores_1 = self.scorenet1(sel_desc) # 시계방향 스코어
        scores_2 = self.scorenet2(sel_desc) # 반시계 스코어
        scores = scores_1 + torch.transpose(scores_2, 1, 2) # 합

        scores = scores_to_permutations(scores) # 스코어 합을 permutation matrix로 만들기 (B,N,N) shape

        return scores
