import torch as th
import torch.nn.functional as F
from sklearn.neighbors import KDTree


def dense_point_generation(model, vox_input, n=10_000, delta=0.009, device='cuda:0'):
    model = model.to(device)

    P_tilde = 2*th.rand(n, 2).to(device) - 1  # [-1, 1]
    P_tilde.requires_grad_()

    P = th.zeros((0, 2))

    latent = model.model.encode(vox_input.unsqueeze(0).to(device))

    i = 0
    while len(P) < n:
        i += 1
        print(f'iter {i}: {len(P)/n:.2%}')
        dist = model.model.decode(P_tilde.unsqueeze(0), latent)[0]
        grad = th.autograd.grad(dist.sum(), P_tilde,
                                retain_graph=False, create_graph=False)[0]
        P_tilde = P_tilde.detach() - dist[..., None]*F.normalize(grad, dim=1)
        P = th.cat([P, P_tilde[dist < delta].detach().cpu()])
        selected_idx = th.randint(0, P_tilde.shape[0], (n,))
        P_tilde = P_tilde[selected_idx] + th.randn(n, 2).to(device)*delta
    return P


def get_normals(model, vox_input, n=10_000, delta=0.009, device='cuda:0'):
    latent = model.model.encode(vox_input.unsqueeze(0).to(device))
    points = dense_point_generation(model, vox_input, n, delta, device)
    n = len(points)
    P = points.to(device).requires_grad_()
    dist = model.model.decode(P.unsqueeze(0).to(device), latent)[0]
    grad = th.autograd.grad(dist.sum(), P,
                            retain_graph=False, create_graph=False)[0].cpu()
    grad = grad/grad.norm(dim=1, keepdim=True)
    tree = KDTree(points)
    idx = tree.query(points, k=30, return_distance=False)
    normals = grad[idx].mean(dim=1)
    normals = normals/normals.norm(dim=1, keepdim=True)
    return points, normals


def roots_along_ray(model, vox_input, ray, delta=0.001, alpha=.1, device='cuda:0'):
    lbd = 0
    p0 = p = ray = ray.to(device)
    latent = model.model.encode(vox_input.unsqueeze(0).to(device))
    dist = th.inf
    i = 0
    while dist > delta:
        i += 1
        print(f'iter {i}')
        p.requires_grad_()
        dist = model.model.decode(p.unsqueeze(0).unsqueeze(0), latent)[0]
        lbd += alpha*dist
        p = p0 + lbd*ray
    return p.detach().cpu()
