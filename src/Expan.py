# modified from CGExpan: https://github.com/yzhan238/CGExpan
import os
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos
import random
import math
from collections import defaultdict as ddict
import queue
import copy
import nltk
import inflect
import pickle
from utils import *
import time

DEFAULT_K = 5
DEFAULT_GEN_THRES = 3
GENERATION_SAMPLE_SIZE = 6
EXPANSION_SAMPLE_SIZE = 3
POS_CNAME_THRES = 5. / 6


class Expan(object):

    def __init__(self, dataset_name, device, model_name='bert-base-uncased', dim=768):

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)

        self.maskedLM = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
        self.maskedLM.to(device)
        self.maskedLM.eval()

        self.k = DEFAULT_K
        self.gen_thres = DEFAULT_GEN_THRES
        self.eid2name, self.name2eid, self.keywords, self.eid2idx = load_vocab(os.path.join(DATA_FOLDER_PATH, dataset_name, 'entity2id.txt'))

        self.entity_pos = pickle.load(open(os.path.join(DATA_FOLDER_PATH, dataset_name, 'entity_pos.pkl'), 'rb'))

        self.pretrained_emb = np.memmap(os.path.join(DATA_FOLDER_PATH, dataset_name, 'pretrained_emb.npy'), dtype='float32', mode='r',
                                        shape=(self.entity_pos[-1], dim))

        tmp = []
        for emb in self.get_emb_iter():
            if len(emb) > 0:
                tmp.append(np.mean(emb, axis=0))
            else:
                tmp.append(np.zeros(len(tmp[0])))
        self.means = np.array(tmp)
        #self.means = np.array([np.mean(emb, axis=0) for emb in self.get_emb_iter()])
        self.inflect = inflect.engine()

        mask_token = self.tokenizer.mask_token
        self.generation_templates2 = [
            [mask_token, ' such as {} and {} .', 1],
            ['such ' + mask_token, ' as {} and {} .', 1],
            ['{} , {} or other ' + mask_token, ' .', 0],
            ['{} , {} and other ' + mask_token, ' .', 0],
            [mask_token, ' including {} and {} .', 1],
            [mask_token, ' , especially {} and {} .', 1],
        ]

        self.ranking_templates = [
            '{} such as ' + mask_token + ' .',
            'such {} as ' + mask_token + ' .',
            mask_token + ' or other {} .',
            mask_token + ' and other {} .',
            '{} including ' + mask_token + ' .',
            '{} especially ' + mask_token + ' .',
        ]

        self.expansion_templates2 = [
            ('', ' such as {} , {} and {} .'),
            ('such ', ' as {} , {} and {} .'),
            ('{} , {} , {} or other ', ' .'),
            ('{} , {} , {} and other ', ' .'),
            ('', ' including {} , {} , and {} .'),
            ('', ' , especially {} , {} , and {} .'),
        ]

        self.generation_templates = [
            [mask_token, ' such as {} , {} , and {} .', 1],
            ['such ' + mask_token, ' as {} , {} , and {} .', 1],
            ['{} , {} , {} or other ' + mask_token, ' .', 0],
            ['{} , {} , {} and other ' + mask_token, ' .', 0],
            [mask_token, ' including {} , {} , and {} .', 1],
            [mask_token, ' , especially {} , {} , and {} .', 1],
        ]

        self.expansion_templates = [
            ('', ' such as {} , {} , {} , and {} .'),
            ('such ', ' as {} , {} , {} , and {} .'),
            ('{} , {} , {} , {} or other ', ' .'),
            ('{} , {} , {} , {} and other ', ' .'),
            ('', ' including {} , {} , {} , and {} .'),
            ('', ' , especially {} , {} , {} , and {} .'),
        ]

        """self.try_templates = [
            '{} , {} and {} often appear in discussions about ' + mask_token + ' .',
            'discussions about ' + mask_token + ' usually contain {} , {} and {} .',
            '{} , {} and {} are very common in discussions about ' + mask_token + ' .',
            'we often mention {} , {} and {} when we discuss ' + mask_token + ' .',
            'when we mention ' + mask_token + ' , it’s hard not to think of {} , {} and {} .',
        ]"""

        self.try_templates = [
            mask_token + ' such as {} , {} and {} .',
            'such ' + mask_token + ' as {} , {} and {} .',
            '{} , {} , {} or other ' + mask_token + ' .',
            '{} , {} , {} and other ' + mask_token + ' .',
            mask_token + ' including {} , {} and {} .',
            mask_token + ' especially {} , {} and {} .',
        ]

        self.calculated_cname_rep = {}

    def rand_idx(self, l):
        for _ in range(10000):
            for i in np.random.permutation(l):
                yield i

    def get_emb(self, i):
        return self.pretrained_emb[self.entity_pos[i]:self.entity_pos[i + 1]]

    def get_emb_iter(self):
        for i in range(len(self.keywords)):
            yield self.pretrained_emb[self.entity_pos[i]:self.entity_pos[i + 1]]

    def expand(self, query_set, target_size, m=2):

        print('start expanding: ' + str([self.eid2name[eid] for eid in query_set]))

        start_time = time.time()

        expanded_set = []

        prev_cn = set()

        neg_set = set()
        neg_cnames = set()
        decrease_count = 0
        margin = m

        while len(expanded_set) < target_size:

            print(f'num of expanded entities: {len(expanded_set)}, time: {int((time.time() - start_time) / 60)} min {int(time.time() - start_time) % 60} sec')
            # print([self.eid2name[eid] for eid in expanded_set])
            # generate class names

            set_text = [self.eid2name[q].lower() for q in query_set + expanded_set]

            if len(set_text) == 2:
                cname2count = self.class_name_generation2(set_text)
            else:
                cname2count = self.class_name_generation(set_text)
            pos_cname, neg_cnames = self.class_name_ranking(cname2count, query_set, expanded_set, neg_cnames, prev_cn,
                                                            margin)
            prev_cn.add(pos_cname)
            # expansion
            if len(set_text) == 2:
                new_entities = self.class_guided_expansion2(pos_cname, query_set + expanded_set, set_text, neg_set)
            else:
                new_entities = self.class_guided_expansion(pos_cname, query_set + expanded_set, set_text, neg_set)
            # filter
            current_expanded_size = len(expanded_set)
            expanded_set.extend(new_entities)
            continue
            #if len(expanded_set) <= target_size // 2: continue
            expanded_set, filter_out = self.class_guided_filter(query_set, expanded_set, pos_cname, neg_cnames,
                                                                cname2count)
            neg_set = neg_set | filter_out
            if len(expanded_set) <= current_expanded_size:
                decrease_count += 1
                if decrease_count >= 2:
                    margin += 1
                    decrease_count = 0
                    neg_set = set()
                    neg_cnames = set()
                    if margin >= 50:
                        break
            else:
                decrease_count = 0

        return expanded_set

    def class_name_generation(self, set_text):
        cname2count = ddict(int)
        idx_generator = self.rand_idx(len(set_text))
        for _ in range(GENERATION_SAMPLE_SIZE):
            for template in self.generation_templates:
                candidate = set()
                q = queue.Queue()
                q.put([])

                template = copy.deepcopy(template)
                indices = []
                for n in idx_generator:
                    if n not in indices:
                        indices.append(n)
                        if len(indices) == 3:
                            break
                template[template[2]] = template[template[2]].format(*[set_text[i] for i in indices])

                while not q.empty():
                    c = q.get()
                    if len(c) >= 2:
                        continue
                    text = template[0] + (' ' if len(c) > 0 else '') + ' '.join(c) + template[1]
                    ids = torch.tensor([self.tokenizer.encode(text, max_length=512, truncation=True)]).long()
                    mask_pos = (ids == self.tokenizer.mask_token_id).nonzero()[0, 1]
                    ids = ids.cuda()
                    with torch.no_grad():
                        predictions = self.maskedLM(ids)[0]
                    _, predicted_index = torch.topk(predictions[0, mask_pos], k=3)
                    predicted_token = [self.tokenizer.convert_ids_to_tokens([idx.item()])[0] for idx in predicted_index]
                    for t in predicted_token:
                        tag = nltk.pos_tag([t] + c)
                        tag = tag[0][1]
                        if tag in set(['JJ', 'NNS', 'NN']) and t not in set(c) \
                                and t not in set([self.inflect.plural(cc) for cc in c]) and t not in ['other', 'such',
                                                                                                      'others']:
                            if len(c) == 0 and tag == 'JJ':
                                continue
                            if len(c) == 0 and tag == 'NN':
                                t = self.inflect.plural(t)
                            new = [t] + c
                            candidate.add(tuple(new))
                            q.put(new)
                for c in candidate:
                    cname2count[' '.join(c)] += 1
        return cname2count

    def class_name_generation2(self, set_text):
        cname2count = ddict(int)
        idx_generator = self.rand_idx(len(set_text))
        for _ in range(GENERATION_SAMPLE_SIZE):
            for template in self.generation_templates2:
                candidate = set()
                q = queue.Queue()
                q.put([])

                template = copy.deepcopy(template)
                indices = []
                for n in idx_generator:
                    if n not in indices:
                        indices.append(n)
                        if len(indices) == 2:
                            break
                template[template[2]] = template[template[2]].format(*[set_text[i] for i in indices])

                while not q.empty():
                    c = q.get()
                    if len(c) >= 2:
                        continue
                    text = template[0] + (' ' if len(c) > 0 else '') + ' '.join(c) + template[1]
                    ids = torch.tensor([self.tokenizer.encode(text, max_length=512, truncation=True)]).long()
                    mask_pos = (ids == self.tokenizer.mask_token_id).nonzero()[0, 1]
                    ids = ids.cuda()
                    with torch.no_grad():
                        predictions = self.maskedLM(ids)[0]
                    _, predicted_index = torch.topk(predictions[0, mask_pos], k=3)
                    predicted_token = [self.tokenizer.convert_ids_to_tokens([idx.item()])[0] for idx in predicted_index]
                    for t in predicted_token:
                        tag = nltk.pos_tag([t] + c)
                        tag = tag[0][1]
                        if tag in set(['JJ', 'NNS', 'NN']) and t not in set(c) \
                                and t not in set([self.inflect.plural(cc) for cc in c]) and t not in ['other', 'such',
                                                                                                      'others']:
                            if len(c) == 0 and tag == 'JJ':
                                continue
                            if len(c) == 0 and tag == 'NN':
                                t = self.inflect.plural(t)
                            new = [t] + c
                            candidate.add(tuple(new))
                            q.put(new)
                for c in candidate:
                    cname2count[' '.join(c)] += 1
        return cname2count

    def class_name_ranking(self, cname2count, query_set, expanded_set, neg_cnames, prev_cn, margin):
        current_set = query_set + expanded_set
        ids = []
        cnames = [cname for cname in cname2count if cname2count[cname] >= self.gen_thres]
        cnames += [cn for cn in prev_cn if cn not in cnames]
        cname2idx = {cname: i for i, cname in enumerate(cnames)}
        cnames_rep = np.vstack([self.get_cname_rep(cname) for cname in cnames])
        scores = np.zeros((len(current_set), len(cnames)))
        for i, eid in enumerate(current_set):
            emb = self.get_emb(self.eid2idx[eid])
            if len(emb) < self.k:
                continue
            sims = cos(cnames_rep, emb)
            for j in range(len(cnames)):
                scores[i, j] = np.mean(np.partition(np.amax(sims[j * 6:(j + 1) * 6], axis=0), -self.k)[-self.k:])
        cname2mrr = ddict(float)
        for eid, score in zip(current_set, scores):
            r = 0.
            for i in np.argsort(-score):
                cname = cnames[i]
                if cname2count[cname] < min(GENERATION_SAMPLE_SIZE * len(self.generation_templates) * POS_CNAME_THRES,
                                            max(cname2count.values())) and cname not in prev_cn:
                    continue
                r += 1
                cname2mrr[cname] += 1 / r
        pos_cname = sorted(cname2mrr.keys(), key=lambda x: cname2mrr[x], reverse=True)[0]

        # find negative entities
        uni_cnames = [cname for cname in cnames if len(cname.split(' ')) == 1 and not pos_cname.endswith(cname)]
        this_neg_cnames = set(uni_cnames)
        for eid, score in zip(query_set, scores):
            ranked_uni_cnames = sorted([pos_cname] + uni_cnames, key=lambda x: score[cname2idx[x]], reverse=True)
            for i, cname in enumerate(ranked_uni_cnames):
                if cname == pos_cname:
                    break
            this_neg_cnames = this_neg_cnames & set(ranked_uni_cnames[i + 1 + margin:])
        return pos_cname, neg_cnames | this_neg_cnames

    def class_guided_expansion(self, pos_cname, current_set, set_text, neg_set):
        global_idx_generator = self.rand_idx(len(current_set))
        local_idx_generator = self.rand_idx(len(current_set))
        global_scores = cos(self.means[[self.eid2idx[eid] for eid in current_set]], self.means)

        ids = []
        for _ in range(EXPANSION_SAMPLE_SIZE):
            for template in self.expansion_templates:
                indices = []
                for n in local_idx_generator:
                    if n not in indices:
                        indices.append(n)
                        if len(indices) == 3:
                            break
                fill_in = [self.tokenizer.mask_token] + [set_text[i] for i in indices]
                fill_in = np.random.permutation(fill_in)
                text = template[0] + pos_cname + template[1]
                text = text.format(*fill_in)
                ids.append(self.tokenizer.encode(text, max_length=512, truncation=True))
        mask_rep = self.get_mask_rep(ids)
        eid2mrr = ddict(float)
        for local_rep in mask_rep:
            indices = []
            for n in global_idx_generator:
                if n not in indices:
                    indices.append(n)
                    if len(indices) == 3:
                        break
            this_global_score = np.mean(global_scores[indices], axis=0)
            this_global_score_ranking = np.argsort(-this_global_score)

            this_keywords = [self.keywords[i] for i in this_global_score_ranking[:500]]
            this_global_score = [this_global_score[i] for i in this_global_score_ranking[:500]]
            this_embs = [self.get_emb(i) for i in [self.eid2idx[eid] for eid in this_keywords]]
            this_entity_pos = [0] + list(np.cumsum([len(emb) for emb in this_embs]))
            this_embs = np.vstack(this_embs)

            raw_local_scores = cos(local_rep[np.newaxis, :], this_embs)[0]

            local_scores = np.zeros((500,))
            for i in range(500):
                start_pos = this_entity_pos[i]
                end_pos = this_entity_pos[i + 1]
                if end_pos - start_pos < self.k:
                    local_scores[i] = 1e-8
                else:
                    local_scores[i] = np.mean(np.partition(raw_local_scores[start_pos:end_pos], -self.k)[-self.k:])

            scores = 5 * np.log(local_scores) + np.log(this_global_score)

            r = 0.
            for i in np.argsort(-scores):
                eid = this_keywords[i]
                if eid not in set(current_set) and eid not in neg_set:
                    r += 1
                    eid2mrr[eid] += 1 / r
                if r >= 20:
                    break

        eid_rank = sorted(eid2mrr, key=lambda x: eid2mrr[x], reverse=True)
        for i, eid in enumerate(eid_rank):
            if eid2mrr[eid] < EXPANSION_SAMPLE_SIZE * len(self.expansion_templates) * 0.2:
                break
        return eid_rank[:max(10, i)]

    def class_guided_expansion2(self, pos_cname, current_set, set_text, neg_set):
        global_idx_generator = self.rand_idx(len(current_set))
        local_idx_generator = self.rand_idx(len(current_set))
        global_scores = cos(self.means[[self.eid2idx[eid] for eid in current_set]], self.means)

        ids = []
        for _ in range(EXPANSION_SAMPLE_SIZE):
            for template in self.expansion_templates2:
                indices = []
                for n in local_idx_generator:
                    if n not in indices:
                        indices.append(n)
                        if len(indices) == 2:
                            break
                fill_in = [self.tokenizer.mask_token] + [set_text[i] for i in indices]
                fill_in = np.random.permutation(fill_in)
                text = template[0] + pos_cname + template[1]
                text = text.format(*fill_in)
                ids.append(self.tokenizer.encode(text, max_length=512, truncation=True))
        mask_rep = self.get_mask_rep(ids)
        eid2mrr = ddict(float)
        for local_rep in mask_rep:
            indices = []
            for n in global_idx_generator:
                if n not in indices:
                    indices.append(n)
                    if len(indices) == 2:
                        break
            this_global_score = np.mean(global_scores[indices], axis=0)
            this_global_score_ranking = np.argsort(-this_global_score)

            this_keywords = [self.keywords[i] for i in this_global_score_ranking[:500]]
            this_global_score = [this_global_score[i] for i in this_global_score_ranking[:500]]
            this_embs = [self.get_emb(i) for i in [self.eid2idx[eid] for eid in this_keywords]]
            this_entity_pos = [0] + list(np.cumsum([len(emb) for emb in this_embs]))
            this_embs = np.vstack(this_embs)

            raw_local_scores = cos(local_rep[np.newaxis, :], this_embs)[0]

            local_scores = np.zeros((500,))
            for i in range(500):
                start_pos = this_entity_pos[i]
                end_pos = this_entity_pos[i + 1]
                if end_pos - start_pos < self.k:
                    local_scores[i] = 1e-8
                else:
                    local_scores[i] = np.mean(np.partition(raw_local_scores[start_pos:end_pos], -self.k)[-self.k:])

            scores = 5 * np.log(local_scores) + np.log(this_global_score)

            r = 0.
            for i in np.argsort(-scores):
                eid = this_keywords[i]
                if eid not in set(current_set) and eid not in neg_set:
                    r += 1
                    eid2mrr[eid] += 1 / r
                if r >= 20:
                    break

        eid_rank = sorted(eid2mrr, key=lambda x: eid2mrr[x], reverse=True)
        for i, eid in enumerate(eid_rank):
            if eid2mrr[eid] < EXPANSION_SAMPLE_SIZE * len(self.expansion_templates2) * 0.2:
                break
        return eid_rank[:max(10, i)]

    def class_guided_filter(self, query_set, expanded_set, pos_cname, neg_cnames, cname2count):

        cnames = [pos_cname] + list(neg_cnames)
        cname2idx = {cname: i for i, cname in enumerate(cnames)}
        cnames_rep = np.vstack([self.get_cname_rep(cname) for cname in cnames])

        filter_out = set()
        for eid in expanded_set:
            emb = self.get_emb(self.eid2idx[eid])
            sims = cos(cnames_rep, emb)
            cnt = 0
            for i in range(len(self.ranking_templates)):
                scores = np.mean(
                    np.partition(sims[[j * 6 + i for j in range(len(cnames))]], -self.k, axis=1)[:, -self.k:], axis=1)
                if np.argmax(scores) != cname2idx[pos_cname]:
                    cnt += 1
            if cnt > 3:
                filter_out.add(eid)
        temp = set([cn for cn in cname2count if
                    cname2count[cn] >= GENERATION_SAMPLE_SIZE * len(self.generation_templates) / 6.])
        temp.update([self.inflect.plural(cn) for cn in temp])
        filter_out.update([eid for eid in expanded_set if self.eid2name[eid].lower() in temp])
        return [eid for eid in expanded_set if eid not in filter_out], filter_out

    def get_cname_rep(self, cname):
        if cname not in self.calculated_cname_rep:
            ids = []
            for template in self.ranking_templates:
                text = copy.deepcopy(template).format(cname)
                ids.append(self.tokenizer.encode(text, max_length=512, truncation=True))
            self.calculated_cname_rep[cname] = self.get_mask_rep(ids)
        return self.calculated_cname_rep[cname]

    def get_mask_rep(self, batch_ids):
        batch_max_length = max(len(ids) for ids in batch_ids)
        ids = torch.tensor([ids + [0 for _ in range(batch_max_length - len(ids))] for ids in batch_ids]).long()
        masks = (ids != 0).long()
        temp = (ids == self.tokenizer.mask_token_id).nonzero()
        mask_pos = []
        for ti, t in enumerate(temp):
            assert t[0].item() == ti
            mask_pos.append(t[1].item())
        ids = ids.to('cuda')
        masks = masks.to('cuda')
        with torch.no_grad():
            batch_final_layer = self.maskedLM(ids, masks)[1][-1]
        return np.array([final_layer[idx].cpu().numpy() for final_layer, idx in zip(batch_final_layer, mask_pos)])

    def get_class_name(self, candidates, hot_words):
        global_idx_generator = self.rand_idx(len(hot_words))
        idx_generator = self.rand_idx(len(hot_words))
        global_scores = cos(self.means[[self.eid2idx[self.name2eid[wd]] for wd in hot_words]], self.means[[self.eid2idx[eid] for eid in candidates]])
        ids = []
        for _ in range(50):
            for template in self.try_templates:
                indices = []
                for n in idx_generator:
                    if n not in indices:
                        indices.append(n)
                        if len(indices) == 3:
                            break
                fill_in = [hot_words[i] for i in indices]
                text = template.format(*fill_in)
                ids.append(self.tokenizer.encode(text, max_length=512, truncation=True))
                """
                id = torch.tensor([self.tokenizer.encode(text, max_length=512)]).long()
                mask_pos = (id == self.tokenizer.mask_token_id).nonzero()[0, 1]
                id = id.cuda()
                with torch.no_grad():
                    predictions = self.maskedLM(id)[0]
                _, predicted_index = torch.topk(predictions[0, mask_pos], k=3)
                predicted_token = [self.tokenizer.convert_ids_to_tokens([idx.item()])[0] for idx in predicted_index]
                for t in predicted_token:
                    print(t, text)
                """
        mask_rep = self.get_mask_rep(ids)
        eid2mrr = ddict(float)
        for local_rep in mask_rep:
            indices = []
            for n in global_idx_generator:
                if n not in indices:
                    indices.append(n)
                    if len(indices) == 3:
                        break

            num = len(candidates)
            this_global_score = np.mean(global_scores[indices], axis=0)
            this_embs = [self.get_emb(i) for i in [self.eid2idx[eid] for eid in candidates]]
            this_entity_pos = [0] + list(np.cumsum([len(emb) for emb in this_embs]))
            this_embs = np.vstack(this_embs)

            raw_local_scores = cos(local_rep[np.newaxis, :], this_embs)[0]

            local_scores = np.zeros((num,))
            for i in range(num):
                start_pos = this_entity_pos[i]
                end_pos = this_entity_pos[i + 1]
                if end_pos - start_pos < self.k:
                    local_scores[i] = 1e-8
                else:
                    local_scores[i] = np.mean(np.partition(raw_local_scores[start_pos:end_pos], -self.k)[-self.k:])

            scores = 5 * np.log(local_scores) + np.log(this_global_score)

            r = 0.
            for i in np.argsort(-scores):
                eid = candidates[i]
                r += 1
                eid2mrr[eid] += 1 / r

        eid_rank = sorted(eid2mrr, key=lambda x: eid2mrr[x], reverse=True)
        for i in range(10):
            print(eid_rank[i], self.eid2name[eid_rank[i]], eid2mrr[eid_rank[i]])
        return eid_rank
