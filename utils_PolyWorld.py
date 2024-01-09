import torch
import numpy as np
from scipy.optimize import linear_sum_assignment # Scipy에 구현된 헝가리안 알고리즘 사용


def scores_to_permutations(scores):
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


def permutations_to_polygons(perm, graph, out='torch'): # graph는 위치가 refine된 vertices positions (B N 2)
    B, N, N = perm.shape
    def bubble_merge(poly):
        s = 0
        P = len(poly)
        while s < P: # s: 0 ~ p-1까지 실행
            head = poly[s][-1]

            t = s+1
            while t < P: # t: s+1 ~ p-1까지 실행
                tail = poly[t][0]
                if head == tail: # 가르키는 좌표가 같아질 때 재귀적으로 호출
                    poly[s] = poly[s] + poly[t][1:]
                    del poly[t]
                    poly = bubble_merge(poly)
                    P = len(poly)
                t += 1
            s += 1
        return poly # (K 2)
    # (0,0),(1,1),...,(n-1,n-1) 대각선 요소에서 값이 0이면 True값, 0이 아닌 값이면 False값 갖는 행렬 (B N)
    # 값이 0이라면 자기 자신이 아닌 다른 요소를 가르킨다. 페어가 생긴다는 뜻
    diag = torch.logical_not(perm[:,range(N),range(N)])
    batch = []
    for b in range(B):
        b_perm = perm[b]
        b_graph = graph[b] # (N, 2)
        b_diag = diag[b]
        
        idx = torch.arange(N)[b_diag] # 값이 0인 대각선 요소의 idx 리스트, 페어를 갖는 노드의 idx 리스트 shape=len(pairs)=idx.shape[0]=(M)

        if idx.shape[0] > 0: # 페어를 갖는 노드가 1개 이상이라면, let idx.shape[0] be M
            # If there are vertices in the batch

            b_perm = b_perm[idx,:] # idx에 해당하는 행만 뽑기 (M, N)
            b_graph = b_graph[idx,:] # idx에 해당하는 행만 뽑기 (M, 2)
            b_perm = b_perm[:,idx] # 뽑아놓은 행에서 idx에 해당하는 열까지 뽑기 (M, M)

            first = torch.arange(idx.shape[0]).unsqueeze(1) # (M 1), 행 정보 리스트
            second = torch.argmax(b_perm, dim=1).unsqueeze(1).cpu() # b_perm에서 최대값이 위치한 col_idx를 모아서 2차원으로 (M 1), 열 정보 리스트

            polygons_idx = torch.cat((first, second), dim=1).tolist() # 행과 열을 합쳐서 x,y 쌍을 갖는다 (M 2)
            polygons_idx = bubble_merge(polygons_idx) # (x,y) 좌표들을 가지고 폴리곤 리턴

            batch_poly = []
            for p_idx in polygons_idx: # 배치 리스트에 저장
                if out == 'torch': # 토치 형태
                    batch_poly.append(b_graph[p_idx,:]) # (M, 2)
                elif out == 'numpy': # 넘파이 형태
                    batch_poly.append(b_graph[p_idx,:].numpy()) # (M, 2)
                elif out == 'list': # 리스트 형태
                    g = b_graph[p_idx,:] * 300 / 320
                    g[:,0] = -g[:,0]
                    g = torch.fliplr(g)
                    batch_poly.append(g.tolist()) # (M, 2)
                elif out == 'coco': # coco 데이터셋 형식에 맞는 형태
                    g = b_graph[p_idx,:] * 300 / 320
                    g = torch.fliplr(g)
                    batch_poly.append(g.view(-1).tolist()) # (M, 2)
                else:
                    print("Indicate a valid output polygon format")
                    exit()
        
            batch.append(batch_poly)

        else:
            # If the batch has no vertices
            batch.append([])

    return batch

