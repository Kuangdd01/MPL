import time
import torch


@torch.no_grad()
def get_mask_similarity_matrix_by_threshold(inner_sim: torch.Tensor,
                                            atten: torch.Tensor, threshold: float, fill_value: float,
                                            device='cpu', ) -> torch.tensor:
    b, k = atten.shape[0], atten.shape[-1]
    eye_mask = torch.eye(b, dtype=bool, device=device).unsqueeze(-1).unsqueeze(-1)
    eye_mask = eye_mask.expand(-1, -1, k, k)
    inner_sim.masked_fill_(eye_mask, 0.)
    bool_mask = (inner_sim >= threshold).to(device=device)  # [b,b,k,k]
    position_tensor = bool_mask.nonzero()  # [nums, 4]
    atten_mask = torch.zeros(atten.shape, dtype=bool, device=device)
    for pos in position_tensor:
        a, b, c, d = pos.tolist()
        atten_mask[a, b, :, d] = True
        atten_mask[b, a, :, c] = True
    atten.masked_fill_(atten_mask, fill_value)
    return atten

@torch.no_grad()
def get_mask_similarity_matrix_by_threshold_(inner_sim: torch.Tensor,
                                             atten: torch.Tensor, threshold: float, fill_value: float,
                                             device='cpu') -> torch.tensor:
    b, k = atten.shape[0], atten.shape[-1]
    eye_mask = torch.eye(b, dtype=bool, device=device).unsqueeze(-1).unsqueeze(-1)
    eye_mask = eye_mask.expand(-1, -1, k, k)
    inner_sim = inner_sim.masked_fill(eye_mask, 0.)
    bool_mask = (inner_sim >= threshold)  # [b, b, k, k]

    pos = bool_mask.nonzero(as_tuple=True)
    atten_mask = torch.zeros(atten.shape, dtype=bool, device=device)
    a, b, c, d = pos
    atten_mask[a, b, :, d] = True
    atten_mask[b, a, :, c] = True
    atten.masked_fill_(atten_mask, fill_value)
    return atten



if __name__ == "__main__":
    pass