# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Sinkhorn(torch.autograd.Function):
    """
    An implementation of a Sinkhorn layer with our custom backward module, based on implicit differentiation
    :param c: input cost matrix, size [*,m,n], where * are arbitrarily many batch dimensions
    :param a: first input marginal, size [*,m] 행에 대해서 합을 1로 맞추기 위한 스케일링용 행렬
    :param b: second input marginal, size [*,n] 열에 대해서 합을 1로 맞추기 위한 스케일링용 행렬 
    :param num_sink: number of Sinkhorn iterations
    :param lambd_sink: entropy regularization weight 행에서 칸마다 코스트의 차이가 작거나 클 경우를 위한 정규화
    :return: optimized soft permutation matrix
    """
    @staticmethod
    def forward(ctx, c, a, b, num_sink, lambd_sink):
        log_p = -c / lambd_sink
        log_a = torch.log(a).unsqueeze(dim=-1)
        log_b = torch.log(b).unsqueeze(dim=-2)
        for _ in range(num_sink):
            log_p -= (torch.logsumexp(log_p, dim=-2, keepdim=True) - log_b)
            log_p -= (torch.logsumexp(log_p, dim=-1, keepdim=True) - log_a)
        p = torch.exp(log_p)

        ctx.save_for_backward(p, torch.sum(p, dim=-1), torch.sum(p, dim=-2))
        ctx.lambd_sink = lambd_sink
        return p

    @staticmethod
    def backward(ctx, grad_p):
        p, a, b = ctx.saved_tensors

        m, n = p.shape[-2:]
        batch_shape = list(p.shape[:-2])

        grad_p *= -1 / ctx.lambd_sink * p
        K = torch.cat((torch.cat((torch.diag_embed(a), p), dim=-1),
                       torch.cat((p.transpose(-2, -1), torch.diag_embed(b)), dim=-1)), dim=-2)[..., :-1, :-1]
        t = torch.cat((grad_p.sum(dim=-1), grad_p[..., :, :-1].sum(dim=-2)), dim=-1).unsqueeze(-1)
        grad_ab, _ = torch.solve(t, K)
        grad_a = grad_ab[..., :m, :]
        grad_b = torch.cat((grad_ab[..., m:, :], torch.zeros(batch_shape + [1, 1], device=DEVICE, dtype=torch.float32)), dim=-2)
        U = grad_a + grad_b.transpose(-2, -1)
        grad_p -= p * U
        grad_a = -ctx.lambd_sink * grad_a.squeeze(dim=-1)
        grad_b = -ctx.lambd_sink * grad_b.squeeze(dim=-1)
        return grad_p, grad_a, grad_b, None, None, None


if __name__ == '__main__':
    c = torch.rand((1, 5, 4)).cuda()
    # tmp = [[[1,7,3],[5,10,4],[7,4,1]]]
    # c = torch.tensor(tmp).cuda()
    a = torch.ones(1, 5).cuda()
    b = torch.ones(1, 4).cuda()
    # a = torch.ones(1, 3).cuda()
    # b = torch.ones(1, 3).cuda()
    # p = Sinkhorn.apply(c, a, b, 100, 1e-2)
    p = Sinkhorn.apply(c, a, b, 100, 0.1)
    print(c)
    print(p.size())
    dims = p.size()
    print(p)
    for b in range(dims[0]):
        for row in range(dims[1]):
            idx = -1
            proba = -1
            for col in range(dims[2]):
                if p[b][row][col] > proba:
                    idx = col
                    proba = p[b][row][col]
            print(row, idx)
    print()
    for b in range(dims[0]):
        for col in range(dims[2]):
            _sum = 0
            for row in range(dims[1]):
                _sum += p[b][row][col]
            print(_sum)
    print()
    for b in range(dims[0]):
        for row in range(dims[1]):
            _sum = 0
            for col in range(dims[2]):
                _sum += p[b][row][col]
            print(_sum)


    
    