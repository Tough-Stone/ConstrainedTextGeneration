import torch
from torch.nn.utils.rnn import pad_sequence


class RankModel(object):

    def __init__(self, device, model, tokenizer, repetition_penalty = 1):
        """
        :param device:
        :param forward_lm: an instance for LSTMLanguageModel, GPT2 LM .
        :param forward_lm_tokenizer:
        """
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.repetition_penalty = repetition_penalty
        self.model.to(self.device)
        self.model.eval()
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def perplexity(self, input_ids=None, input_texts=None):
        print(f'input_ids={input_ids}')
        if input_ids is None:
            assert input_texts is not None
            input_ids = []
            for text in input_texts:
                ids = self.tokenizer.encode(text)
                ids = [self.tokenizer.bos_token_id] + ids + [self.tokenizer.eos_token_id]
                input_ids.append(torch.tensor(ids))

        label_ids = [s.clone() for s in input_ids]
        print(f'label_ids={label_ids}')
        lengths_tensors = torch.tensor([len(s) - 1 for s in input_ids])
        print(f'lengths_tensors={lengths_tensors}')
        # gpt2 does not have the [PAD] token.
        # pad input with 0 (the padded value can be arbitrary number.)
        input_tensors = pad_sequence(input_ids, batch_first=True, padding_value=0)
        print(f'input_tensors={input_tensors}')
        # pad label with -100 (can not be other number.)
        labels_tensors = pad_sequence(label_ids, batch_first=True, padding_value=-100)
        print(f'labels_tensors={labels_tensors}')
        # 1 for real tokens and 0 for padded tokens
        masks_tensors = torch.zeros(labels_tensors.shape, dtype=torch.float32)
        print(f'masks_tensors={masks_tensors}')
        masks_tensors = masks_tensors.to(self.device)
        input_tensors = input_tensors.to(self.device)
        labels_tensors = labels_tensors.to(self.device)
        lengths_tensors = lengths_tensors.to(self.device)
        masks_tensors = masks_tensors.masked_fill(labels_tensors != -100, 1)
        labels_tensors = labels_tensors[:, 1:]

        outputs = self.model(input_tensors, attention_mask=masks_tensors)
        logits = outputs[0]
        logits = logits[:, :-1, :]
        loss_ = self.loss_func(logits.reshape(-1, logits.shape[-1]), labels_tensors.reshape(-1))
        loss_ = loss_.reshape(labels_tensors.shape)
        loss_ = torch.sum(loss_, dim=-1).double()
        # log_ppls = (loss_ / lengths_tensors).cpu().numpy()
        # probs = torch.exp(-loss_).cpu().numpy()
        log_ppls = (loss_ / lengths_tensors)
        probs = torch.exp(-loss_)
        return log_ppls, probs