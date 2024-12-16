import torch
from typing import List
from einops import rearrange, repeat


@torch.no_grad()
def get_inner_similarity(region_feature: torch.Tensor, temperature):
    # norm backbone features
    region_feature = region_feature / (region_feature.norm(dim=-1, keepdim=True) + 
                                       torch.finfo(region_feature.dtype).eps)
    inner_sim = torch.einsum('b k d, a q d -> b a k q',region_feature,region_feature)
    inner_sim /= temperature
    inner_sim_k = inner_sim.max(dim=-1)[0] #[b a k]
    inner_sim_b = inner_sim_k.mean(dim=-1) #[b a]
    score = torch.softmax(inner_sim_b, dim=-1) #[b,b]
    return score

def image_io(image_id, preprocess, device="cpu"):
    from PIL import Image
    tensor_list = []
    flickr_image_root = ""
    image_path_list = [flickr_image_root + str(img) + ".jpg" for img in image_id]
    for path in image_path_list:
        image = Image.open(path).convert('RGB')
        transform_image = preprocess(image)
        tensor_list.append(transform_image)
    image_tensor = torch.stack(tensor_list).to(device=device)
    return image_tensor

@torch.no_grad()
def get_inner_similarity_by_unciom(image_id: List, device="cpu") -> torch.Tensor:
    import unicom
    model, preprocess = unicom.load("ViT-B/16")
    model.to(device=device)
    input_image_tensors = image_io(image_id, preprocess, device)
    assert input_image_tensors.device == next(model.parameters()).device
    unicom_out = model(input_image_tensors) #[B,DIM]
    sim = torch.einsum('b d, a d -> b a', unicom_out, unicom_out)
    score = torch.softmax(sim, dim=-1)
    return score


# @torch.no_grad()
def get_mask_similarity_matrix_by_threshold(inner_sim: torch.Tensor,
                                            atten: torch.Tensor, threshold: float, fill_value: float,
                                              device='cpu', x_mask_plus = None)-> torch.tensor:
    atten_back = atten.clone()
    b, k = atten.shape[0], atten.shape[-1]
    eye_mask = torch.eye(b,dtype=bool,device=device).unsqueeze(-1).unsqueeze(-1)
    eye_mask = eye_mask.expand(-1, -1, k, k)
    inner_sim.masked_fill_(eye_mask, 0.)
    bool_mask = (inner_sim >= threshold).to(device=device) #[b,b,k,k]

    
    pos = bool_mask.nonzero(as_tuple=True)
    atten_mask = torch.zeros(atten.shape, dtype=bool, device=device)
    a, b, c, d = pos
    atten_mask[a, b, :10, d] = True
    atten_mask[b, a, :10, c] = True

    atten.masked_fill_(atten_mask, fill_value)
    return atten, atten_back, atten_mask

    
def converting(atten_mask: torch.Tensor, origin_atten: torch.Tensor, 
               diag_logits: torch.Tensor, device=None):
    
    def log_softmax(x, dim=-1):
        maxv = x.amax(dim=dim, keepdim=True)
        x = x - maxv
        x = x - torch.logsumexp(x, dim=dim, keepdim=True)
        return x
    b, a, q, k = origin_atten.shape

    masked_atten = atten_mask * origin_atten #[b b q k]
    eye_mask = torch.eye(b,dtype=bool,device=device).unsqueeze(-1).unsqueeze(-1)
    eye_mask = eye_mask.expand(-1, -1, q, k) #[b b q k]
    diag_atten = eye_mask * origin_atten
    merge_atten = masked_atten + diag_atten
    # ipdb.set_trace()
    merge_atten_logits = torch.where(merge_atten == 0, torch.tensor(float('-inf')).to(device), merge_atten)
    merge_atten_logits_ = rearrange(merge_atten_logits, 'b a q k -> b q (a k)',b=b, q=q, a=a, k=k)
    merge_atten_weight = merge_atten_logits_.softmax(dim=-1)
    merge_atten_weight = rearrange(merge_atten_weight, 'b q (a k) ->b a q k',b=b, q=q, a=a, k=k)
    cvt_weight = merge_atten_weight * atten_mask # [b a q k]

    reshape_origin_att = rearrange(origin_atten, 'b a q k -> b q (a k)',b=b, q=q, a=a, k=k)

    msk_output = log_softmax(reshape_origin_att, dim=-1)
    
    msk_output = rearrange(msk_output, 'b q (a k) -> b a q k',b=b, q=q, a=a, k=k)
    converted_loss = msk_output * cvt_weight
    
    c_loss = converted_loss.sum(dim=2).sum() * -1  
    return c_loss
