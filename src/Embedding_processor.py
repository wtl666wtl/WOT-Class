import os
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm
import argparse
import pickle
from utils import *

def get_pretrained_emb(model, tokenizer, sentences, entity_pos, eid2idx, np_file, dim=768, batch_size=64):
    fp = np.memmap(np_file, dtype='float32', mode='w+', shape=(entity_pos[-1], dim))
    ptr_list = [0 for _ in entity_pos[:-1]]
    iterations = int(len(sentences)/batch_size) + (0 if len(sentences) % batch_size == 0 else 1)
    print(iterations)
    for i in tqdm(range(iterations)):
        start = i * batch_size
        end = min((i+1)*batch_size, len(sentences))
        batch_ids = []
        for _, sent in sentences[start:end]:
            ids = tokenizer.encode(sent, max_length=512, truncation=True)
            batch_ids.append(ids)
        batch_max_length = max(len(ids) for ids in batch_ids)
        ids = torch.tensor([ids + [0 for _ in range(batch_max_length - len(ids))] for ids in batch_ids]).long()
        masks = (ids != 0).long()
        temp = (ids == tokenizer.mask_token_id).nonzero()
        mask_pos = []
        for ti, t in enumerate(temp):
            assert t[0].item() == ti
            mask_pos.append(t[1].item())
        ids = ids.to('cuda')
        masks = masks.to('cuda')
        with torch.no_grad():
            batch_final_layer = model(ids, masks)[0]
        for final_layer, mask, (eid, _) in zip(batch_final_layer, mask_pos, sentences[start:end]):
            rep = final_layer[mask].cpu().numpy()
            this_idx = entity_pos[eid2idx[eid]] + ptr_list[eid2idx[eid]]
            ptr_list[eid2idx[eid]] += 1
            fp[this_idx] = rep.astype(np.float32)
    del fp


MAX_SENT_LEN = 300


def get_masked_sentences(filename, mask_token, eid2idx):
    sentences = []
    total_line = get_num_lines(filename)
    entity_num = [0 for _ in eid2idx]
    with open(filename, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_line):
            obj = json.loads(line)
            if len(obj['entityMentions']) == 0 or len(obj['tokens']) > MAX_SENT_LEN:
                continue
            raw_sent = [token.lower() for token in obj['tokens']]
            for entity in obj['entityMentions']:
                eid = entity['entityId']
                if eid not in eid2idx:
                    continue
                entity_num[eid2idx[eid]] += 1
                sent = copy.deepcopy(raw_sent)
                sent[entity['start']:entity['end']+1] = [mask_token]
                sentences.append((eid, sent))
    print(f'Sentences: {len(sentences)} masked sentences constructed')
    entity_pos = np.cumsum(entity_num)
    entity_pos = [0] + list(entity_pos)
    return sentences, entity_pos


def load_vocab(filename):
    eid2name = {}
    keywords = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            temp = line.strip().split('\t')
            eid = int(temp[1])
            eid2name[eid] = temp[0]
            keywords.append(eid)
    eid2idx = {w:i for i, w in enumerate(keywords)}
    print(f'Vocabulary: {len(keywords)} keywords loaded')
    return eid2name, keywords, eid2idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='path to dataset folder')
    parser.add_argument('--vocab', default='entity2id.txt', help='vocab file')
    parser.add_argument('--sent', default='sentences.json', help='sent file')
    parser.add_argument('--npy_out', default='pretrained_emb.npy', help='name of output npy file')
    parser.add_argument('--entity_pos_out', default='entity_pos.pkl', help='name of output entity index file')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print('CUDA not available')
        exit()
    print("Loading...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = False)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model.eval()
    print("Wait a minute...")
    train_dataset_name = f"{args.dataset}_train"
    _, _, eid2idx = load_vocab(os.path.join(DATA_FOLDER_PATH, train_dataset_name, args.vocab))
    sentences, entity_pos = get_masked_sentences(os.path.join(DATA_FOLDER_PATH, train_dataset_name, args.sent), tokenizer.mask_token, eid2idx)
    print("Start...")
    pickle.dump(entity_pos, open(os.path.join(DATA_FOLDER_PATH, train_dataset_name, args.entity_pos_out), 'wb'))
    get_pretrained_emb(model, tokenizer, sentences, entity_pos, eid2idx, np_file=os.path.join(DATA_FOLDER_PATH, train_dataset_name, args.npy_out))

