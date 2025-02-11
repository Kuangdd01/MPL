import torch
from einops import rearrange, repeat
from torch.nn import functional as F
def get_duplicate_mask(image_id, output, device='cpu'):
    """
    image_id: id list of batch image
    output: 

    """
    b = output.shape[0]
    q = output.shape[2]
    k = output.shape[-1]
    image_batch = torch.zeros((b,b))
    mask_list = []
    for s, img_idx in enumerate(image_id):
        tmp_list = [0 for _ in range(image_batch.shape[0])]
        for i, j in enumerate(image_id):
            if s != i and img_idx == j:
                tmp_list[i] = 1
        mask_list.append(tmp_list)
    
    mask = torch.tensor(mask_list,dtype=bool, device=device)
    # print(mask_list)
    mask = repeat(mask, 'b a -> b a q k',b=b,a=b, q=q, k=k)
    # print(mask.shape)
    mask = rearrange(mask, 'b a q k -> b q (a k)',b=b,a=b, q=q, k=k)
    bk = b * k
    assert  (b, q, bk) == mask.shape
    return mask

def get_same_label_mask(label_feature: torch.Tensor,
                        label_id: torch.Tensor, 
                        query_embeddings: torch.Tensor, 
                        device='cpu') -> torch.Tensor:
    """
    label_feature: label feature output from model text_encoder
    label_id: data in batch
    query_embeddings: query feature output from model text_encoder
    region_embeddings: region feature form MLP after frozen backbone
    """
    b, q, k = label_feature.shape[0], query_embeddings.shape[1], label_id.shape[1]
    q2l_att = torch.einsum('x q d, y k d -> x y q k',query_embeddings, label_feature)
    q2l_att = torch.einsum('i i q k -> i q k', q2l_att)
    label_pos = F.one_hot(q2l_att.argmax(dim=-1), q2l_att.shape[-1]).float() #[b q k] mask
    assert q2l_att.shape[0], label_feature.shape[1] == label_id.shape  #shape[b k]
    q2label_id = torch.einsum('b q k, b k c -> b q c',label_pos,label_id.unsqueeze(dim=-1).float()).squeeze(-1)
    #test
    # q2label_id = torch.tensor([[3, 1],[4, 3],[5, 2]])
    assert  b, q == q2label_id.shape
    query2label = q2label_id.repeat_interleave(b,dim=0).\
        repeat_interleave(k,dim=1).unsqueeze(-1).view(b*b, q, k).view(b, b, q, k)
    region2label = label_id.repeat(b, q).unsqueeze(-1).view(b*b,k,q).view(b, b, q, k)
    mask = torch.zeros([b,b,q,k],dtype=bool, device=label_feature.device)
    mask = query2label == region2label
    eye_mask = torch.eye(b,dtype=bool,device=device).unsqueeze(-1).unsqueeze(-1)
    eye_mask = eye_mask.expand(-1, -1, q, k)
    mask.masked_fill_(eye_mask, False)
    mask = rearrange(mask, 'b1 b2 q k -> b1 q (b2 k)',b1=b, b2=b, q=q, k=k)
    return mask

@torch.no_grad()
def get_visualization_masked_label_query(mask: torch.Tensor, x_mask: torch.Tensor, phrase_mask: torch.Tensor,
                                         bbox: torch.Tensor, phrases, label_id, idx2word, image_id: torch.Tensor) -> dict:
    """
    get detailed information about the mask position
    Args:
        mask:[b q bk]
        x_mask: [b q k]
    Returns:
    """
    assert len(mask.shape) == 3
    # print("mask mean number for every sentence query: {}".format(mask.sum(dim=-1).mean(dim=-1)))
    print("mask number:{}".format(mask.sum() / phrase_mask.sum()))
    mask = rearrange(mask, 'b q (a k) -> b a q k', b=mask.shape[0], q=mask.shape[1], k=x_mask.shape[-1], a=mask.shape[0])
    valid_mask = mask * x_mask.unsqueeze(0)
    print("valid mask number:{}".format(valid_mask.sum() / phrase_mask.sum()))
    # [b q bk]
    valid_mask = rearrange(valid_mask, 'b a q k -> b q (a k)', b=mask.shape[0], q=x_mask.shape[1], k=x_mask.shape[-1])
    # position_tensor = valid_mask.nonzero()
    label_id = rearrange(label_id, 'b k -> (b k)', b=label_id.shape[0], k=label_id.shape[1])
    bbox = rearrange(bbox, 'b k c -> (b k) c', b=bbox.shape[0], k=bbox.shape[1], c=bbox.shape[2])
    image_id_ = image_id.repeat_interleave(x_mask.shape[-1])
    # print(image_id.shape)
    assert image_id_.shape[0] == mask.shape[0] * x_mask.shape[-1]

    return_list = {}
    for i, batch_tensor in enumerate(valid_mask):
        return_list[i] = {}
        for j, query_tensor in enumerate(batch_tensor):
            if phrase_mask[i][j]:
                position_tensor = query_tensor.nonzero()  # [bk] ->[s,1] box:[bk,4]
                if position_tensor.shape[0] == 0:
                    continue
                bbox_tensor = bbox[position_tensor].squeeze(1)  # [s,4]
                box_image_id = image_id_[position_tensor].squeeze(-1)
                assert bbox_tensor.shape[0] == box_image_id.shape[0]

                label = label_id[position_tensor[0]]
                return_list[i][phrases[i][j]['phrase']] = {
                    'box_info' : [(int(ix.cpu()), box.cpu().tolist()) for ix, box in zip(box_image_id, bbox_tensor)],
                    'label': idx2word[label.item()],
                    'image': int(image_id[i].item())
                }
                # for ix, box in zip(box_image_id, bbox_tensor):
                #     idx, boxx = int(ix.cpu()), box.cpu().tolist()
                #     # real_phrase = phrases[i][j]['phrase']
                #     return_list[i][phrases[i][j]['phrase']] = {
                #         ''
                #     }
                # bbox_list.append(bbox_tensor.cpu().tolist())
                # assert position_tensor.shape[0], 4 == bbox_tensor.shape #可能为空
                # print(position_tensor.shape, bbox_tensor.shape)
                
        # print(return_list[i])
    return return_list

