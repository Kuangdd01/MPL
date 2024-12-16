import functools
import os
import re

import torch
from torch.utils.data import Dataset
import pickle

from tqdm import tqdm
from transformers import PreTrainedTokenizer
from ._glove_tokenizer import GloveTokenizer
from ._mat_image_features_reader import MatImageFeaturesH5Reader
from .flickr30k_entities_utils import get_annotations, get_sentence_data


# ==================================================================================================================== #
#                                                                                                                      #
# ==================================================================================================================== #


class MatFlickrGroundingDataset(Dataset):
    def __init__(
            self,
            dataroot: str,
            split: str,
            image_features_reader: MatImageFeaturesH5Reader,
            tokenizer: PreTrainedTokenizer,
            max_region_num: int = 100,
            max_seq_length: int = 128,
            max_phrase_num: int = 32,
            max_phrase_len: int = 12,
    ):
        self.split = split
        self.image_features_reader = image_features_reader
        self.tokenizer = tokenizer
        self.max_region_num = max_region_num
        self.max_seq_length = max_seq_length
        self.max_phrase_num = max_phrase_num
        self.max_phrase_len = max_phrase_len
        self.last_word_pat = re.compile("[ ,]")

        os.makedirs(os.path.join(dataroot, 'cache'), exist_ok=True)
        self.entries = self.load_annotations(dataroot)
        num_phrase, num_obj = [], []
        for entry in self.entries:
            num_phrase.append(
                min(len(entry['phrases']), self.max_phrase_num)
            )
            num_obj.append(
                self.image_features_reader.get_num_boxes(entry["image_id"])
            )
        self.num_phrase = torch.tensor(num_phrase, dtype=torch.long)
        self.num_obj = torch.tensor(num_obj, dtype=torch.long)

    def load_annotations(self, dataroot):
        entries_cache_path = os.path.join(dataroot, 'cache', f'FlickrGroundingDataset_{self.split}_entry.pkl')
        if not os.path.exists(entries_cache_path):
            entries = self._load_annotations(dataroot)
            pickle.dump(entries, open(entries_cache_path, 'wb'))
        else:
            entries = pickle.load(open(entries_cache_path, 'rb'))
        return entries

    def _load_annotations(self, dataroot):
        entries = []
        with open(os.path.join(dataroot, f'{self.split}.txt'), 'r') as f:
            images = f.read().splitlines()
        for img in tqdm(images, desc=f'[build {self.split} dataset] loading annotations'):
            ann = get_annotations(os.path.join(dataroot, 'Annotations', img + '.xml'))
            sens = get_sentence_data(os.path.join(dataroot, 'Sentences', img + '.txt'))
            for sent in sens:
                entry = {
                    'image_id': int(img),
                    'caption': sent['sentence'],
                    'phrases': [],
                    'refBoxes': [],
                }
                for phrase in sent['phrases']:
                    if str(phrase['phrase_id']) in ann['boxes'].keys():
                        entry['phrases'].append(phrase)
                        entry['refBoxes'].append(ann['boxes'][str(phrase['phrase_id'])])
                assert len(entry['phrases']) <= self.max_phrase_num
                if entry['phrases']:
                    entries.append(entry)
        return entries

    def __len__(self):
        return len(self.entries)

    @functools.lru_cache(maxsize=4000)
    def tokenize_obj_or_attr(self, words):
        wd = self.last_word_pat.split(words)[-1]
        return self.tokenizer.get_vocab().get(wd, self.tokenizer.unk_token_id)

    def _tokenize(self, entry):
        span = []
        caption_encoding = self.tokenizer(entry['caption'].lower().split(),
                                          is_split_into_words=True,
                                          max_length=self.max_seq_length,
                                          padding='max_length',
                                          truncation=True,
                                          return_tensors='pt')
        for phrase in entry['phrases']:
            first_word_index = phrase['first_word_index']
            word_count = len(phrase['phrase'].split())
            if type(self.tokenizer) != GloveTokenizer:
                st, ed = self.tokenizer.word_to_tokens(first_word_index)
                if word_count != 1:
                    _, ed = self.tokenizer.word_to_tokens(first_word_index + word_count - 1)
            else:
                st, ed = caption_encoding.word_to_tokens(first_word_index)
                if word_count != 1:
                    _, ed = caption_encoding.word_to_tokens(first_word_index + word_count - 1)
            span.append((st, ed))
        return caption_encoding, span

    def __getitem__(self, index):
        entry = self.entries[index]

        # pad image features to max_region_num
        image_id = entry["image_id"]
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
        encoding, spans = self._tokenize(entry)
        caption_ids = torch.tensor(encoding.ids)
        length = torch.tensor(sum(encoding.attention_mask), dtype=torch.long)
        span_mask = torch.zeros(self.max_phrase_num, self.max_seq_length, dtype=torch.long)
        for i, (st, ed) in enumerate(spans):
            span_mask[i][st:ed] = 1

        # caption
        caption = entry['caption']

        # phrase span.
        phrases = entry['phrases']

        num_phrase = min(len(phrases), self.max_phrase_num)

        mix_phrases_id = torch.zeros(self.max_phrase_num, self.max_phrase_len, dtype=torch.long)
        for i, phrase in enumerate(phrases):
            words = phrase['phrase'].lower().split(' ')
            mix_phrases_id[i] = torch.tensor(self.tokenizer(words, max_length=self.max_phrase_len).ids, dtype=torch.long)

        # ref_box
        ref_boxes = entry["refBoxes"]
        union_ref_boxes = torch.zeros(self.max_phrase_num, 4)
        for i, boxes in enumerate(ref_boxes):
            union_ref_boxes[i] = torch.tensor(union(boxes))

        return (torch.tensor(index, dtype=torch.long),
                image_id, mix_features_pad, mix_objs_id, torch.tensor(mix_num_boxes, dtype=torch.long),
                caption, caption_ids, span_mask, length,
                phrases, mix_phrases_id, torch.tensor(num_phrase, dtype=torch.long),
                ref_boxes, boxes_ori, union_ref_boxes, boxes_ori_tensor,
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


def union(bboxes):
    leftmin, topmin, rightmax, bottommax = 999, 999, 0, 0
    for box in bboxes:
        left, top, right, bottom = box
        if left == 0 and top == 0 and right == 0 and bottom == 0:
            continue
        leftmin, topmin, rightmax, bottommax = min(left, leftmin), min(top, topmin), max(right, rightmax), max(bottom, bottommax)

    return [leftmin, topmin, rightmax, bottommax]

if __name__ == "__main__":
    dataset = MatFlickrGroundingDataset(split="test")
