import torch
import torch.nn as nn
import numpy as np

from torch.nn import TransformerEncoderLayer, TransformerEncoder


B = 8 # batch size
N = 256 # num of nodes, n_peaks
D = 64 # num of depth, descriptor_dim
DD = 768 # D', dim of init and final desc

v = torch.randn(B, N, 2) # detected nodes v; B, N, 2(x와 y 좌표)
d = torch.randn(B, N, D) # visual descriptor d; B, N, D(dim)

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

        # x를 [B*N, input_dim]로 변환
        x = x.view(B * N, -1)

        # 첫 번째 완전 연결 층을 통과한 후 BatchNorm 적용
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 두 번째 완전 연결 층을 통과
        x = self.fc2(x)

        # x를 원래 형태로 되돌림: [B, N, output_dim]
        x = x.view(B, N, -1)
        return x
    
# 폴리 월드 구현체
class ScoreNet(nn.Module): # 가능한 모든 노드 페어를 계산하고 총 점수를 최대화한다
    def __init__(self, in_ch): # 768 -> 256 -> 128 -> 64 -> 1
        # 768 -> 3072 -> 384 -> 48 -> 1 은 어떨지 궁금
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x): # x: B N D'
        n_points = x.shape[1] # N

        x = x.transpose(1, 2) # B D' N

        x = x.unsqueeze(-1) # B D' N 1
        x = x.repeat(1,1,1,n_points) # B D' N N
        t = torch.transpose(x, 2, 3) # B D' N N 대각선 뒤집기
        x = torch.cat((x, t), dim=1) # B 2D' N N

        x = self.conv1(x) # B 256 N N
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x) # B 128 N N
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x) # B 64 N N
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x) # B 1 N N
        return x[:,0]
    
class ConnectionNet(nn.Module):
    def __init__(self):
        super(ConnectionNet, self).__init__()
        # 인코더 레이어 정의
        encoder_layer = TransformerEncoderLayer(
            d_model=768, 
            nhead=12,
        )

        # 인코더 정의
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, 
            num_layers=2
        )
    
    def forward(self, d_init):
        d_final = self.transformer_encoder(d_init)
        return d_final

class DiG_generator(nn.Module):
    def __init__(self):
        super(DiG_generator, self).__init__()
        # 폴리에서 128->128과 128->64를 사용함
        # TopDiG 논문도 레이어 2개 쓴다고 함
        # 여기서는 66->256과 256->768로 설정해봄
        self.mlp = MLP(D+2, 256, DD)
        self.connectionNet = ConnectionNet()

        # Two parallel graph heads
        self.graphHead1 = ScoreNet(DD * 2)
        self.graphHead2 = ScoreNet(DD * 2)
        

    def forward(self, detected_node, visual_descriptor):
        # B, N, _ = detected_node.shape
        # B, N, D = visual_descriptor.shape

        # 1. it firstly concatenates each coupled d_i and v_i to embedded descriptors d_emb; N D+2.
        d_emb = torch.cat((visual_descriptor, detected_node), dim=2) # d_emb; B N D+2
        # print("d_emb:", d_emb.shape) # 64, 256, 66

        # 2. A MLP is then utilized to encode the d_emb and
        # produce the D' dimensional initial descriptors d_init; B N D'.
        d_init = self.mlp(d_emb) # d_init; B N D'
        # print("d_init:", d_init.shape) # 64 256 768

        # 3. d_init is fed into a connection network that consists of M(2) transformer encoder layers
        # to yield final descriptors d_final; N D'.
        d_final = self.connectionNet(d_init) # d_final; N D'
        # print("d_final:", d_final.shape) # 64 256 768

        # 4. Two Graph heads; they receive the d_final and predict two adj_graphs A and B.
        scores_1 = self.graphHead1(d_final) # B N N
        scores_2 = self.graphHead2(d_final) # B N N

        return scores_1, scores_2
    


if __name__ == "__main__":
    dig = DiG_generator()
    dig.eval()
    s1, s2 = dig(v, d)
    print(s1.shape)
    print(s1.unique())
    print(s2.shape)
    print(s2.unique())