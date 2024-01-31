import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# B = 8  # batch size
# N = 256  # num of nodes, n_peaks
# D = 64  # num of depth, descriptor_dim
# DD = 768  # D', dim of init and final desc

# v = torch.randn(B, N, 2)  # detected nodes v; B, N, 2(x와 y 좌표)
# d = torch.randn(B, N, D)  # visual descriptor d; B, N, D(dim)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x의 크기: [B, N, input_dim]
        B, N, _ = x.shape

        x = x.view(B * N, -1)
        # 첫 번째 완전 연결 층을 통과한 후 BatchNorm 적용
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x를 원래 형태로 되돌림: [B, N, output_dim]
        x = x.view(B, N, -1)
        return x


class ScoreNet(nn.Module):  # 가능한 모든 노드 페어를 계산하고 총 점수를 최대화한다
    def __init__(self, in_ch):  # 768 -> 256 -> 128 -> 64 -> 1
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch,
                               256,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256,
                               128,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,
                               64,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,
                               1,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True)

    def forward(self, x):  # x: B N D'
        n_points = x.shape[1]  # N

        x = x.transpose(1, 2)  # B D' N

        x = x.unsqueeze(-1)  # B D' N 1
        x = x.repeat(1, 1, 1, n_points)  # B D' N N
        t = torch.transpose(x, 2, 3)  # B D' N N 대각선 뒤집기
        x = torch.cat((x, t), dim=1)  # B 2D' N N

        x = self.conv1(x)  # B 256 N N
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)  # B 128 N N
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)  # B 64 N N
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)  # B 1 N N
        return x[:, 0]


class ConnectionNet(nn.Module):
    def __init__(self):
        super(ConnectionNet, self).__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=768,
            nhead=12,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                                      num_layers=2)

    def forward(self, d_init):
        d_final = self.transformer_encoder(d_init)
        return d_final


class DiG_generator(nn.Module):
    def __init__(self, descriptor_dim=64, features=768):
        super(DiG_generator, self).__init__()
        # 폴리에서 128->128과 128->64를 사용함
        # TopDiG 논문도 레이어 2개 쓴다고 함
        # 여기서는 66->256과 256->768로 설정해봄
        self.mlp = MLP(descriptor_dim + 2, 256, features)
        self.connectionNet = ConnectionNet()
        self.graphHead1 = ScoreNet(features * 2)
        self.graphHead2 = ScoreNet(features * 2)

    def forward(self, detected_node, visual_descriptor):
        # B, N, _ = detected_node.shape
        # B, N, D = visual_descriptor.shape

        # B N D+2 (64 256 66)
        d_emb = torch.cat((visual_descriptor, detected_node), dim=2)

        # B N D' (64 256 768)
        d_init = self.mlp(d_emb)

        # B N D' (64 256 768)
        d_final = self.connectionNet(d_init)

        scores_1 = self.graphHead1(d_final)  # B N N
        scores_2 = self.graphHead2(d_final)  # B N N

        return scores_1, scores_2
