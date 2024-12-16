import os
import torch

_writer = None


def set_writer(w):
    global _writer
    if _writer is None:
        _writer = w
    return


def breakpoint_if_find_debug_file():
    if os.path.exists('~/debug.txt'):
        breakpoint()


def breakpoint_if_nan(tensor):
    if isinstance(tensor, torch.Tensor):
        if torch.any(torch.isnan(tensor)):
            breakpoint()


def breakpoint_if_nan_or_inf(tensor):
    if isinstance(tensor, torch.Tensor):
        if torch.any(torch.isnan(tensor)):
            breakpoint()
        if torch.any(torch.isinf(tensor)):
            breakpoint()


def grad_norm_p(parameters):
    total_norm = 0
    filtered_parameters = [p for p in parameters if p.grad is not None and p.requires_grad]
    for p in filtered_parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def writer(func_name, *args, **kwargs):
    getattr(_writer, func_name)(*args, **kwargs)


def writer_grad_norm(tensor, subgroup, name, dim=-1, group='[grad norm]', global_step=None):
    if not isinstance(tensor, torch.Tensor):
        return
    if not tensor.requires_grad:
        return
    if dim is None:
        def hook_fn(grad):
            writer('add_scalars',
                   group, {
                       f"{subgroup} _ {name}": torch.abs(grad).mean()
                   },
                   global_step=global_step,
                   )
    else:
        def hook_fn(grad):
            writer('add_scalars',
                   group, {
                       f"{subgroup} _ {name}": torch.norm(grad, dim=dim).mean()
                   },
                   global_step=global_step,
                   )

    tensor.register_hook(hook_fn)


def writer_var_norm(tensor, subgroup, name, dim=-1, group='[var norm]', global_step=None):
    if not isinstance(tensor, torch.Tensor):
        return
    if dim is None:
        writer('add_scalars',
               group, {
                   f"{subgroup} _ {name}": torch.abs(tensor).mean()
               },
               global_step=global_step,
               )
    else:
        writer('add_scalars',
               group, {
                   f"{subgroup} _ {name}": torch.norm(tensor, dim=dim).mean()
               },
               global_step=global_step,
               )
        
def writer_region_inner_similarity(tensor, subgroup, name, dim=-1, group='[region inner similarity]', global_step=None,
                                   region_mask=None, image_id=None):
    if not isinstance(tensor, torch.Tensor):
        return
    region_mask_ = region_mask
    region_mask = region_mask.unsqueeze(-1).expand(tensor.size())
    tensor = tensor * region_mask
    tensor = tensor / (tensor.norm(dim=dim, keepdim=True) + torch.finfo(tensor.dtype).eps)
    sim = torch.einsum('b k d, a q d -> b a k q', tensor, tensor)  # [b,b,k,k]
    sim_region_score = sim.max(dim=-1)[0].sum(dim=-1)  # [b,b,k]->[b,b]
    diag = torch.eye(sim.size(1), device=tensor.device)
    non_diag = (~diag.bool()).int()
    sim_ = sim_region_score * non_diag
    # 对于某一张最匹配的图片 其中最值为1？
    score = sim_.max(dim=-1)[0].mean()

    writer('add_scalars',
           group, {
               f"{subgroup} _ {name} _ inner_sim": score
           },
           global_step=global_step, )

    image_id_list = image_id
    from collections import Counter
    counter = Counter(image_id_list)
    repeat = 0
    for k, v in counter.items():
        if v > 1:
            repeat += 1
    writer('add_scalars',
           group, {
               f"{subgroup} _ {name} _ repeat": repeat
           },
           global_step=global_step, )
    
    # write mean region number
    writer('add_scalars',
           group, {
               f"batch _ {name} _ mean_region_nums": region_mask_.sum(-1).float().mean()
           },
           global_step=global_step, )

def len2mask(length, shape):
    batch, max_length = shape
    mask = torch.lt(torch.arange(max_length, device=length.device).unsqueeze(0).expand(shape), length.unsqueeze(1))
    return mask
