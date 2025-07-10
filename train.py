import torch
import torch.nn.functional as F
from utils import CONFIG, estimate_shapley

def train_local_step(encoder, surrogate, x, z, optimizer_enc, optimizer_sur, alpha, device):
    encoder.train()
    surrogate.train()

    x = x.to(device)
    z = z.to(device)

    with torch.no_grad():
        h = encoder(x)
    pred_z = surrogate(h.detach())
    loss_privacy = F.cross_entropy(pred_z, z)
    optimizer_sur.zero_grad()
    loss_privacy.backward()
    optimizer_sur.step()

    h = encoder(x)
    pred_z = surrogate(h)
    privacy_loss = F.cross_entropy(pred_z, z)
    loss_enc = -alpha * privacy_loss

    optimizer_enc.zero_grad()
    loss_enc.backward()
    optimizer_enc.step()

    return loss_enc.item(), loss_privacy.item(), h.detach()


def train_server_step(server_head, embeddings, y, optimizer_server, device):
    server_head.train()
    y = y.to(device)
    concat_h = torch.cat(embeddings, dim=1).to(device)
    preds = server_head(concat_h)
    loss = F.cross_entropy(preds, y)
    optimizer_server.zero_grad()
    loss.backward()
    optimizer_server.step()
    return loss.item()


@torch.no_grad()
def evaluate_attack_accuracy(encoder, surrogate, dataloader, client_id, device):
    encoder.eval()
    surrogate.eval()
    correct = 0
    total = 0
    for (x1, x2), _, (z1, z2) in dataloader:
        x = x1 if client_id == 0 else x2
        z = z1 if client_id == 0 else z2
        x = x.to(device)
        z = z.to(device)
        h = encoder(x)
        preds = surrogate(h).argmax(dim=1)
        correct += (preds == z).sum().item()
        total += z.size(0)
    return correct / total


@torch.no_grad()
def evaluate_target_accuracy(encoders, server_head, dataloader, device):
    for model in encoders:
        model.eval()
    server_head.eval()

    correct = 0
    total = 0
    for (x1, x2), y, _ in dataloader:
        h1 = encoders[0](x1.to(device))
        h2 = encoders[1](x2.to(device))
        h_cat = torch.cat([h1, h2], dim=1)
        preds = server_head(h_cat).argmax(dim=1)
        correct += (preds == y.to(device)).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def compute_shapley_q(encoders, server_head, dataloader, device):
    server_head.eval()
    h_cache = {}
    y_cache = None
    for (x1, x2), y, _ in dataloader:
        h_cache[0] = encoders[0](x1.to(device))
        h_cache[1] = encoders[1](x2.to(device))
        y_cache = y.to(device)
        break

    utility = {}
    for subset in [(0,), (1,), (0, 1), ()]:
        if not subset:
            utility[subset] = 0.0
            continue
        h_subset = [h_cache[k] for k in subset]
        h_cat = torch.cat(h_subset, dim=1)
        preds = server_head(h_cat)
        u = 1.0 - F.cross_entropy(preds, y_cache).item()
        utility[subset] = u

    return estimate_shapley(utility, total_clients=2)
