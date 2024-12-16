import functools
import re

import numpy as np

import torch
from einops import rearrange
from torch.utils.data import Dataset
from tqdm import tqdm

from ._volta_refer_image_features_reader import RefImageFeaturesH5Reader
from tools.refer.refer import REFER


@functools.lru_cache()
def get_refer(*args, **kwargs):
    return REFER(*args, **kwargs)


class ReferExpressionDataset(Dataset):
    def __init__(self,
                 dataroot,
                 name,
                 split,
                 image_features_reader,
                 tokenizer,
                 max_region_num: int = 40,
                 max_seq_length: int = 40,
                 ):
        super().__init__()
        self.dataroot = dataroot
        self.name = name
        self.split = split
        self.image_features_reader = image_features_reader
        self.tokenizer = tokenizer
        self.max_region_num = max_region_num
        self.max_seq_length = max_seq_length
        self.max_phrase_num = 1  # compatible with Phrase grounding
        self.max_phrase_len = max_seq_length  # compatible with Phrase grounding
        self.last_word_pat = re.compile("[ ,]")
        #
        self.refer = get_refer(dataroot, name)
        self.ref_ids = self.refer.getRefIds(split=split)
        self.entries = self.load_annotations()
        #
        num_obj = []
        for entry in tqdm(self.entries, 'build num obj'):
            num_obj.append(
                self.image_features_reader.get_num_boxes(entry['image_id'])
            )
        self.num_phrase = torch.ones((len(self.entries),), dtype=torch.long)
        self.num_obj = torch.tensor(num_obj, dtype=torch.long)

    def load_annotations(self):
        entries = []
        for ref_id in tqdm(self.ref_ids, 'load annotations'):
            ref = self.refer.loadRefs(ref_id)[0]
            unformatted_box = self.refer.getRefBox(ref_id)
            box = [unformatted_box[0],
                   unformatted_box[1],
                   unformatted_box[0] + unformatted_box[2],
                   unformatted_box[1] + unformatted_box[3]]
            image_id = ref['image_id']
            for sentence in ref['sentences']:
                sent = sentence['sent']
                tokens = sentence['tokens']
                sent_id = sentence['sent_id']
                entries.append({
                    "ref_id": ref_id,
                    "sent_id": sent_id,
                    "image_id": image_id,
                    "sent": sent,
                    "tokens": tokens,
                    "box": box,
                })
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]

        # pad image features to max_region_num
        image_id = entry['image_id']
        features, num_boxes, _, boxes_ori, objs_words, _ = self.image_features_reader[image_id]

        mix_num_boxes = min(int(num_boxes), self.max_region_num)
        # obj feature
        mix_features_pad = torch.zeros((self.max_region_num, self.image_features_reader.feature_size))
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]
        # boxes_ori
        boxes_ori_tensor = torch.zeros(self.max_region_num, 4)
        boxes_ori_tensor[:mix_num_boxes] = torch.tensor(boxes_ori[:mix_num_boxes])

        # obj attr words
        objs_id = [self.tokenize_obj_or_attr(w) for w in objs_words]
        mix_objs_id = torch.zeros(self.max_region_num, dtype=torch.long)
        mix_objs_id[:mix_num_boxes] = torch.tensor(objs_id[:mix_num_boxes], dtype=torch.long)

        # tokenize
        encoding = self._tokenize(entry)
        token_ids = torch.tensor(encoding.ids)
        span_mask = rearrange(torch.tensor(encoding.attention_mask, dtype=torch.long), "(n d) -> n d", n=1)
        length = torch.sum(span_mask)
        num_phrase = 1

        # phrase
        phrases = entry['sent']
        mix_phrases_id = rearrange(token_ids, '(n d) -> n d', n=1)

        # ref_box
        ref_boxes = torch.tensor(entry["box"])
        union_ref_boxes = rearrange(ref_boxes, '(n d) -> n d', n=1)
        return (torch.tensor(index, dtype=torch.long),
                image_id, mix_features_pad, mix_objs_id, torch.tensor(mix_num_boxes, dtype=torch.long),
                entry, token_ids, span_mask, length,
                phrases, mix_phrases_id, torch.tensor(num_phrase, dtype=torch.long),
                entry, boxes_ori, union_ref_boxes, boxes_ori_tensor,
                )

    def collate_fn(self, batch):
        (index,
         image_id, mix_features_pad, mix_objs_id, mix_num_boxes,
         caption, caption_ids, span_mask, length,
         phrases, mix_phrases_id, num_phrase,
         ref_boxes, boxes_ori, union_ref_boxes, boxes_ori_tensor,
         ) = zip(*batch)

        should_stack = (index,
                        mix_features_pad, mix_objs_id, mix_num_boxes,
                        caption_ids, span_mask, length,
                        mix_phrases_id, num_phrase,
                        union_ref_boxes, boxes_ori_tensor,
                        )
        stacked = map(lambda x: torch.stack(x), should_stack)
        (index,
         mix_features_pad, mix_objs_id, mix_num_boxes,
         caption_ids, span_mask, length,
         mix_phrases_id, num_phrase,
         union_ref_boxes, boxes_ori_tensor,
         ) = stacked

        return (index,
                image_id, mix_features_pad, mix_objs_id, mix_num_boxes,
                caption, caption_ids, span_mask, length,
                phrases, mix_phrases_id, num_phrase,
                ref_boxes, boxes_ori, union_ref_boxes, boxes_ori_tensor,
                )

    @functools.lru_cache(maxsize=4000)
    def tokenize_obj_or_attr(self, words):
        wd = self.last_word_pat.split(words)[-1]
        return self.tokenizer.get_vocab().get(wd, self.tokenizer.unk_token_id)

    def _tokenize(self, entry):
        caption_encoding = self.tokenizer(entry['tokens'],
                                          is_split_into_words=True,
                                          max_length=self.max_seq_length,
                                          padding='max_length',
                                          truncation=True,
                                          return_tensors='pt')
        return caption_encoding
