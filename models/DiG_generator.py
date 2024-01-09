import torch
import torch.nn as nn
import numpy as np

from torch.nn import TransformerEncoderLayer, TransformerEncoder
from scipy.optimize import linear_sum_assignment # Scipy에 구현된 헝가리안 알고리즘 사용


B = 8 # batch size
N = 256 # num of nodes, n_peaks
D = 64 # num of depth, descriptor_dim
DD = 768 # D', dim of init and final desc

# 트레이너에 필요한 인풋
# adjacency matrix label; B, N, N
label_mat = torch.randn(B, N, N)

# 필요한 인풋
# detected nodes v; B, N, 2(x와y 좌표)
v = torch.randn(B, N, 2)
# visual descriptor d; B, N, D(dim)
d = torch.randn(B, N, D)

# def MLP(channels: list, batch_norm=True): # 폴리월드의 MLP구현체
#     n_layers = len(channels) # 채널 수

#     layers = []
#     for i in range(1, n_layers): # 1 ~ 채널 수 - 1
#         layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True)) # (0,1),(1,2),...,(n-2,n-1)

#         if i < (n_layers - 1): # 마지막 레이러 빼고 배치놈+렐루 추가하기
#             if batch_norm:
#                 layers.append(nn.BatchNorm1d(channels[i]))
#             layers.append(nn.ReLU())

#     return nn.Sequential(*layers)
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

def scores_to_permutations(scores): # 인퍼런스용 함수
    """
    Input a batched array of scores and returns the hungarian optimized 
    permutation matrices.
    """
    B, N, N = scores.shape

    scores = scores.detach().cpu().numpy()
    perm = np.zeros_like(scores)
    for b in range(B):
        # sinkhorn 알고리즘은 비공개된 트레인 파일에 있을 듯
        # 반복적인 노멀라이제이션(100번)을 위해 싱크홀 알고리즘 사용
        # 인퍼런스시에 헝가리안 알고리즘으로 linear sum assignment result 뽑는다
        r, c = linear_sum_assignment(-scores[b]) # 점수가 높을 수록 페어일 확률이 높으므로 -를 붙여서 최소 찾는 문제로 바꾼다.
        perm[b,r,c] = 1 # 헝가리안 알고리즘이 찾은 칸은 1로 아니면 0인 permutation matrix (B N N) 만든다
    return torch.tensor(perm) # 텐서로 바꿔주기


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
        # print("detected_node:", type(detected_node)) # detected_node는 튜플
        # print("visual_descriptor:", visual_descriptor.shape)
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

        # 4. Two Graph heads; they receive the d_final and predict two adj_graphs A; N N and B; N N.
        # 시간이 오래 걸리는데 cuda로 디바이스 변경해서 실행해봐야 할 듯
        scores_1 = self.graphHead1(d_final) # B N N
        # print("scores_1: ", scores_1.shape)
        scores_2 = self.graphHead2(d_final) # B N N
        # print("scores_2: ", scores_2.shape)

        # 5. add up to export the A_final
        scores = scores_1 + torch.transpose(scores_2, 1, 2)
        # print(scores.unique())

        # 6. Sinkhorn 알고리즘 적용 for polygon-shape target TODO 이거는 시각화 하기위해 좌표 찍는 과정
        # 어차피 트레인 시에는 scores 행렬로 loss 계산할거기 때문에
        # predicted_adjacency_matrix = scores_to_permutations(scores) # 지금은 헝가리안 적용됨, matcher_sinkhorn.py 수정하여 적용할수도
        # 위 한 줄은 인퍼런스용임. 훈련용은 scores 리턴하면 된다.
        # 6-2.아웃풋 poly형태로 갈거면 한 줄 더 추가
        # poly = permutations_to_polygons(scores, graph, out='coco') # matrix 폴리곤으로 만들기


        # 7. loss_graph: binary cross-entropy loss로 supervised한다. p: A와 B, p_hat: inputs
        # trainer.py 만들고 코드 짜서 loss = BCSLoss() 걸고 트레인하면 될듯

        # scores는 한칸 한칸 -무한대~무한대의 범위를 갖는 score로 이루어져 있다
        # loss_fn 안에서 softmax로 범위 조정 필요함
        return scores
    


if __name__ == "__main__":
    dig = DiG_generator()
    dig.eval()
    out = dig(v, d)
    print(out.shape)
    print(out.unique())