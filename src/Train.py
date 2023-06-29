import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from pympler import asizeof
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BartForTextInfill, AdamW
from utils.Log import Logger


class BARTDataset(Dataset):
    def __init__(self, mode, max_sentence_length=40, encoder_loss_type=0,
                 statistics=False, local_rank=-2, ratio1=0.5, ratio2=0.5):
        self.encoder_loss_type = encoder_loss_type
        assert mode in ["train", "test", 'valid']
        self.mode = mode
        if self.mode == 'test' or self.mode == 'valid':
            self.is_train = False
        else:
            self.is_train = True
        self.tokenizer = tokenizer
        self.max_sentence_length = max_sentence_length + 2  # the bos and eos tokens
        self.encoder_inputs = []
        self.encoder_labels = []
        self.decoder_labels = []
        self.mask_labels = []

        data_dict_path = f'../dataset/synthetic/{mode}_segment3_synthetic.json'
        # data_dict_path = f'../dataset/synthetic/ttt.json'
        if os.path.exists(data_dict_path):
            print(f'''加载数据......(from {data_dict_path})''')
            with open(data_dict_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                self.encoder_inputs += content['encoder_input_list']
                self.encoder_labels += content['encoder_label_list']
                self.decoder_labels += content['decoder_label_list']

                for j in range(len(self.encoder_labels)):
                    mask_label = []
                    index = 0  # 记录decoder_label中的下标
                    for i in range(len(self.encoder_labels[j])):
                        if self.encoder_labels[j][i] == 0:
                            mask_label.append(self.encoder_inputs[j][i])
                            index += 1
                        elif self.encoder_labels[j][i] == 1:
                            mask_label.append([tokenizer.pad_token_id] * (len(self.decoder_labels[j][index])))
                            index += 1
                        else:
                            mask_label.append([tokenizer.pad_token_id] * len(self.decoder_labels[j][index]))
                            mask_label.append(self.encoder_inputs[j][i])
                            index += 2
                    self.mask_labels.append(mask_label)

                encoder_inputs_trans_list = []
                encoder_labels_trans_list = []
                decoder_labels_trans_list = []
                mask_labels_trans_list = []
                for i in range(len(self.encoder_inputs)):
                    encoder_inputs_trans = []
                    encoder_labels_trans = []
                    decoder_labels_trans = []
                    mask_labels_trans = []
                    for j in range(len(self.encoder_inputs[i])):  # 每个input中的每个关键词token
                        if self.encoder_labels[i][j] == 0:
                            for k in range(len(self.encoder_inputs[i][j])):
                                encoder_labels_trans.append(0)
                        elif self.encoder_labels[i][j] == 1:
                            for k in range(len(self.encoder_inputs[i][j])):
                                encoder_labels_trans.append(1)
                        else:
                            for k in range(len(self.encoder_inputs[i][j])):
                                if k == 0:
                                    encoder_labels_trans.append(2)
                                else:
                                    encoder_labels_trans.append(0)
                        for k in range(len(self.encoder_inputs[i][j])):  # 每个关键词token中的每个字token
                            encoder_inputs_trans.append(self.encoder_inputs[i][j][k])
                    encoder_inputs_trans_list.append(encoder_inputs_trans)
                    encoder_labels_trans_list.append(encoder_labels_trans)
                    for j in range(len(self.decoder_labels[i])):
                        for k in range(len(self.decoder_labels[i][j])):
                            decoder_labels_trans.append(self.decoder_labels[i][j][k])
                            mask_labels_trans.append(self.mask_labels[i][j][k])
                    decoder_labels_trans_list.append(decoder_labels_trans)
                    mask_labels_trans_list.append(mask_labels_trans)

                self.encoder_inputs = encoder_inputs_trans_list
                self.encoder_labels = encoder_labels_trans_list
                self.decoder_labels = decoder_labels_trans_list
                self.mask_labels = mask_labels_trans_list


        else:
            print(f'请先创建合成数据：{data_dict_path}。')

        self.len = len(self.encoder_inputs)

        if statistics and local_rank in [-1, 0]:
            print('Statistics for sentence length:')
            lengths = [len(e) for e in self.decoder_labels]
            (unique, counts) = np.unique(lengths, return_counts=True)
            for k, v in zip(unique, counts):
                print(f'sentence length{k}: {v}')
            print('Statistics for sentence labels:')

            labels = [e for s in self.encoder_labels for e in s]
            (unique, counts) = np.unique(labels, return_counts=True)
            for k, v in zip(unique, counts):
                print(f'Label {k}: {v}')

    def __getitem__(self, idx):
        return torch.tensor(self.encoder_inputs[idx], dtype=torch.long), \
               torch.tensor(self.encoder_labels[idx], dtype=torch.long), \
               torch.tensor(self.decoder_labels[idx], dtype=torch.long), \
               torch.tensor(self.mask_labels[idx], dtype=torch.long)

    def __len__(self):
        return self.len

    def create_mini_batch(self, samples):
        encoder_inputs = [s[0] for s in samples]
        encoder_labels = [s[1] for s in samples]
        decoder_labels = [s[2] for s in samples]
        decoder_inputs = [s[3] for s in samples]

        # Mask to avoid performing attention on padding token indices in encoder_inputs.
        _mask = pad_sequence(encoder_inputs, batch_first=True, padding_value=-100)
        attention_mask = torch.zeros(_mask.shape, dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)

        # for i in range(len(encoder_inputs)):
        #     print(len(encoder_inputs[i]), len(encoder_labels[i]), len(decoder_labels[i]))
        # print('==========')
        # for i in range(len(encoder_inputs)):
        #     print(encoder_inputs[i])
        #     print(encoder_labels[i])
        #     print(decoder_labels[i])
        # print('==========')

        # this method is for non-autoregressive decoding.
        # decoder_inputs = [self.create_decoder_inputs(s[0], s[1], tokenizer.mask_token_id) for s in samples]

        # finals = []
        #
        # for k in range(len(decoder_inputs)):
        #     final = []
        #     start_index = 0
        #     decoder_labels[k] = decoder_labels[k].numpy().tolist()
        #     for i in range(len(decoder_inputs[k])):
        #         if decoder_inputs[k][i - 1] != 103 and decoder_inputs[k][i] == 103:
        #             left = i - 1
        #             right = i + 1
        #             while decoder_inputs[k][left] == 103:
        #                 left -= 1
        #             while decoder_inputs[k][right] == 103:
        #                 right += 1
        #             num = decoder_labels[k].index(decoder_inputs[k][right], decoder_labels[k].index(decoder_inputs[k][left], start_index)+1)\
        #                   - decoder_labels[k].index(decoder_inputs[k][left], start_index)
        #             start_index = decoder_labels[k].index(decoder_inputs[k][right], decoder_labels[k].index(decoder_inputs[k][left], start_index))
        #             for j in range(num - 1):
        #                 final.append(tokenizer.mask_token_id)
        #         elif decoder_inputs[k][i - 1] == 103 and decoder_inputs[k][i] == 103:
        #             continue
        #         else:
        #             final.append(decoder_inputs[k][i].numpy().tolist())
        #     final = torch.Tensor(final).to(torch.int64)
        #     finals.append(final)
        #
        # decoder_inputs = finals

        encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        encoder_labels = pad_sequence(encoder_labels, batch_first=True, padding_value=-100)
        if self.encoder_loss_type == 1:  # labels for mse loss
            encoder_labels = encoder_labels.float()

        decoder_labels = pad_sequence(decoder_labels, batch_first=True, padding_value=-100)
        # avoid computing loss on the first token, i.e. bos_token
        decoder_labels[:, 0] = -100

        # replace the eos_token_id with pad_token_id
        for i, _ in enumerate(decoder_inputs):
            decoder_inputs[i][-1] = self.tokenizer.pad_token_id

        decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # create decoder_inputs by shifting the decoder_labels right,
        _tmp = decoder_inputs.clone()
        decoder_inputs[:, 1:] = _tmp[:, :-1]
        decoder_inputs[:, 0] = self.tokenizer.eos_token_id

        # construct labels for masked lm loss
        masked_lm_labels = decoder_labels.clone()
        masked_lm_labels[_tmp != tokenizer.mask_token_id] = -100

        return encoder_inputs, encoder_labels, decoder_inputs, decoder_labels, masked_lm_labels, attention_mask

    @staticmethod
    def create_decoder_inputs(encoder_inputs, encoder_labels, mask_token_id):
        """
        :param encoder_inputs: list, each element is an int
        :param encoder_labels: list, each element is an int
        :return:
        """
        decoder_inputs = []
        for i, l in zip(encoder_inputs, encoder_labels):
            if l == 0:
                decoder_inputs.append(i)
            elif l == 1:
                decoder_inputs.append(mask_token_id)
            else:
                decoder_inputs += [mask_token_id] * (l - 1)
                decoder_inputs.append(i)
        return torch.tensor(decoder_inputs, dtype=torch.long)

    @staticmethod
    def compute_accuracy(model, mode, local_rank, dataloader, datasize, device, num_labels, encoder_loss_type,
                         merge_insert=False, ):
        """
        compute negative log-likelihood on dataloader with model.
        :param model:
        :param dataloader:
        :return:
        """
        if local_rank in [-1, 0]:
            print()
        model.eval()
        correct = {}
        recalls = {}
        precisions = {}
        f1s = {}
        for i in range(num_labels):
            recalls[i] = 0.0
            precisions[i] = 0.0
            correct[i] = 0.0
        total_encoder_loss = 0
        total_decoder_loss = 0
        total_masked_decoder_loss = 0
        with torch.no_grad():
            start = time.time()
            step = 0
            for data in dataloader:
                data = [t.to(device) for t in data]
                encoder_inputs, encoder_labels, decoder_inputs, decoder_labels, masked_lm_labels, attention_mask = data
                encoder_loss, decoder_loss, encoder_logits, logits = model(encoder_inputs,
                                                                           encoder_labels=encoder_labels,
                                                                           decoder_input_ids=decoder_inputs,
                                                                           labels=decoder_labels,
                                                                           attention_mask=attention_mask)[:4]
                bts = encoder_inputs.shape[0]
                total_encoder_loss += encoder_loss * bts
                total_decoder_loss += decoder_loss * bts

                loss_fct = torch.nn.CrossEntropyLoss()
                # only compute labels for mask tokens
                total_masked_decoder_loss += loss_fct(logits.view(-1, logits.shape[-1]),
                                                      masked_lm_labels.view(-1)) * bts

                # compute accuracy
                if encoder_loss_type == 0:  # classification
                    # argmax
                    predict_label = torch.argmax(encoder_logits, dim=-1, keepdim=False)
                else:  # regression, round and convert the output into torch.Long tensor
                    predict_label = torch.round(encoder_logits).long()

                if merge_insert:
                    predict_label[predict_label > 2] = 2
                    encoder_labels[encoder_labels > 2] = 2

                for i in range(num_labels):
                    correct[i] += ((predict_label == i) & (encoder_labels == i)).sum()
                    recalls[i] += (encoder_labels == i).sum()
                    precisions[i] += ((predict_label == i) & (encoder_labels != -100)).sum()

                step += bts
                if local_rank in [-1, 0]:
                    print(
                        f'\r{mode} set {step}/{datasize}, time: {time.time() - start:.1f} seconds.',
                        end='')
                # if step>=100:
                # break
            # if torch.cuda.device_count() > 1:
            #     torch.distributed.all_reduce_multigpu([total_encoder_loss])
            #     torch.distributed.all_reduce_multigpu([total_decoder_loss])
            #     torch.distributed.all_reduce_multigpu([total_masked_decoder_loss])
            total_encoder_loss = total_encoder_loss.item()
            total_decoder_loss = total_decoder_loss.item()
            total_masked_decoder_loss = total_masked_decoder_loss.item()

            total_loss = total_encoder_loss + total_decoder_loss
            average_encoder_loss = total_encoder_loss / datasize
            average_decoder_loss = total_decoder_loss / datasize
            average_masked_decoder_loss = total_masked_decoder_loss / datasize
            average_loss = total_loss / datasize

            # merge results
            for i in range(num_labels):
                # if torch.cuda.device_count() > 1:
                #     torch.distributed.all_reduce_multigpu([correct[i]])
                #     torch.distributed.all_reduce_multigpu([recalls[i]])
                #     torch.distributed.all_reduce_multigpu([precisions[i]])
                correct[i] = correct[i].item()
                recalls[i] = recalls[i].item()
                precisions[i] = precisions[i].item()

            for i in range(num_labels):
                if recalls[i] != 0:
                    recalls[i] = correct[i] / recalls[i]
                else:
                    recalls[i] = 0

                if precisions[i] != 0:
                    precisions[i] = correct[i] / precisions[i]
                else:
                    precisions[i] = 0

                if precisions[i] != 0:
                    f1s[i] = 2 * recalls[i] * precisions[i] / (recalls[i] + precisions[i])
                else:
                    f1s[i] = 0

            used_time = time.time() - start
        model.train()
        return average_encoder_loss, average_decoder_loss, average_masked_decoder_loss, average_loss, \
               used_time, recalls, precisions, f1s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text infilling.")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--test_batch_size', type=int, default=10)
    parser.add_argument('--num_labels', type=int, default=3,
                        help='0 for copy, 1 for replace, 2-5 means insert 1-4 tokens')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--w', type=float, default=1.0, help='The weight for the encoder loss')
    parser.add_argument('--masked_lm', type=float, default=0, help='0 for using language modeling for the decoder,'
                                                                   '1 for using mask language modeling for the decoder.')
    parser.add_argument('--full_mask', type=float, default=0, help='0 for using casual mask attention for decoder, '
                                                                   '1 for without using casual mask attention for decoder.')
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--encoder_loss_type', type=int, default=0,
                        help='0 is classification loss, 1 is regression loss')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--ratio1', type=float, default=0.5)
    parser.add_argument('--ratio2', type=float, default=0.5)
    args = parser.parse_args()
    print(f'可用gpu数量为：', torch.cuda.device_count())
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # 设置控制台参数
    # args.n_gpu = torch.cuda.device_count()  # gpu数量
    args.n_gpu = 1
    model_path = f'../checkpoints/bart-large-chinese'  # 预训练模型路径
    log_path = f'../log'  # 日志文件路径
    if args.masked_lm == 0:
        masked_lm = ''
    else:
        masked_lm = '_masked_lm'
    if args.full_mask == 0:
        full_mask = ''
    else:
        full_mask = '_full_mask'
    if args.local_rank in [-1, 0]:  # 创建日志文件
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = f'{log_path}/train_segment_large.log'
        logger = Logger(log_file)
        logger.logger.info(f'日志文件路径：{log_file}')
        logger.logger.info(args)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logger.logger.info(f'使用{args.n_gpu}个gpu训练模型。')
    if args.local_rank == -1 or args.n_gpu <= 1:  # 指定设备
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
    else:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        print('local_rank:', args.local_rank)
    # 加载预训练模型（bart-chinese）和分词器（bert）
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BartForTextInfill.from_pretrained(model_path, num_labels=args.num_labels,
                                              encoder_loss_type=args.encoder_loss_type)
    model = model.to(device)
    if args.local_rank in [-1, 0]:
        logger.logger.info('初始化BartForTextInfill（from checkpoint {}）'.format(model_path))
    # 准备训练集
    train_set = BARTDataset(mode="train", encoder_loss_type=args.encoder_loss_type,
                            local_rank=args.local_rank, ratio1=args.ratio1, ratio2=args.ratio2)
    if args.local_rank in [-1, 0]:
        print(f'RAM memory size for train set: {asizeof.asizeof(train_set) / (1024.0 ** 3):.2f}G.')
    if args.local_rank in [-1, 0]:
        logger.logger.info(f'训练集加载完成，大小为：{len(train_set)}.')
    if args.local_rank == -1 or args.n_gpu <= 1:
        train_sampler = torch.utils.data.RandomSampler(train_set)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = DataLoader(train_set, num_workers=0, batch_size=args.batch_size,
                              sampler=train_sampler, collate_fn=train_set.create_mini_batch)
    print(f'len(train_set)={len(train_set)}, len(train_loader)={len(train_loader)}')
    # 准备验证集
    valid_set = BARTDataset(mode='valid', encoder_loss_type=args.encoder_loss_type,
                            local_rank=args.local_rank, ratio1=args.ratio1, ratio2=args.ratio2)
    if args.local_rank in [-1, 0]:
        print(f'RAM memory size  for valid set: {asizeof.asizeof(valid_set) / (1024.0 ** 3):.2f}G.')
    if args.local_rank in [-1, 0]:
        logger.logger.info(f'''验证集加载完成，大小为：{len(valid_set)}.''')
    if args.local_rank == -1 or args.n_gpu <= 1:
        test_sampler = torch.utils.data.SequentialSampler(valid_set)
    else:
        test_sampler = torch.utils.data.distributed.DistributedSampler(valid_set)
    valid_loader = DataLoader(valid_set, num_workers=0, batch_size=args.test_batch_size,
                              sampler=test_sampler, collate_fn=valid_set.create_mini_batch)
    print(f'len(valid_set)={len(valid_set)}, len(valid_loader)={len(valid_loader)}')
    # 计算验证集的初始loss
    average_encoder_loss, average_decoder_loss, average_masked_decoder_loss, average_loss, used_time, recalls, precisions, f1s = \
        BARTDataset.compute_accuracy(model, 'valid', args.local_rank, valid_loader, len(valid_set), device,
                                     args.num_labels, encoder_loss_type=args.encoder_loss_type)
    if args.local_rank in [-1, 0]:
        logs = f'\n   valid set, ave loss {average_loss:.3f}, encoder loss {average_encoder_loss:.3f}, decoder loss {average_decoder_loss:.3f},' \
               f' mask decoder loss {average_masked_decoder_loss:.3f}, uses {used_time:.1f} seconds.'
        Macro_P = np.mean(list(precisions.values()))
        Macro_R = np.mean(list(recalls.values()))
        Macro_F1 = np.mean(list(f1s.values()))
        for i in range(len(f1s)):
            logs += f'''\n      Label_{i}: Precision={precisions[i]:.3f},  Recall={recalls[i]:.3f}, F1:{f1s[i]:.3f};'''
        logs += f'''\n      Macro_P={Macro_P:.3f},  Macro_R={Macro_R:.3f}, Macro_F1={Macro_F1:.3f}.'''
        logger.logger.info(logs)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True,
                                                           min_lr=1e-6)
    scheduler.step(average_decoder_loss)
    if args.masked_lm:
        best_loss = average_encoder_loss + average_masked_decoder_loss
    else:
        best_loss = average_encoder_loss + average_decoder_loss
    # 开始训练
    evaluate_steps = max(int(len(train_set) / args.batch_size / 5), 1)
    print_steps = 10
    global_steps = 0
    local_step = 0
    start = time.time()
    total_loss = 0
    total_encoder_loss = 0
    total_decoder_loss = 0
    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader):
            global_steps += 1
            local_step += 1
            data = [t.to(device) for t in data]
            encoder_inputs, encoder_labels, decoder_inputs, decoder_labels, masked_lm_labels, attention_mask = data
            if args.masked_lm:
                decoder_labels = masked_lm_labels
            encoder_loss, decoder_loss, encoder_logits, logits = model(encoder_inputs, encoder_labels=encoder_labels,
                                                                       decoder_input_ids=decoder_inputs,
                                                                       labels=decoder_labels,
                                                                       attention_mask=attention_mask)[:4]
            # zero the parameter gradients
            optimizer.zero_grad()
            # backward
            loss = args.w * encoder_loss + decoder_loss
            # loss =decoder_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_encoder_loss += encoder_loss.item()
            total_decoder_loss += decoder_loss.item()
            if global_steps % print_steps == 0 and args.local_rank in [-1, 0]:
                print(
                    "\rEpoch {}/{}, {}/{}, global steps {}, average loss is {:.3f},  average encoder loss is {:.3f}, "
                    "average decoder loss is {:.3f}, "
                    " {} steps uses {:.1f} seconds.".format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                            global_steps, total_loss / local_step,
                                                            total_encoder_loss / local_step,
                                                            total_decoder_loss / local_step, local_step,
                                                            time.time() - start), end='')
            if global_steps % evaluate_steps == 0:
                average_encoder_loss, average_decoder_loss, average_masked_decoder_loss, average_loss, used_time, recalls, precisions, f1s = \
                    BARTDataset.compute_accuracy(model, 'valid', args.local_rank, valid_loader, len(valid_set), device,
                                                 args.num_labels,
                                                 encoder_loss_type=args.encoder_loss_type)
                if args.local_rank in [-1, 0]:
                    logs = f'\n   valid set, ave loss {average_loss:.3f}, encoder loss {average_encoder_loss:.3f},' \
                           f' decoder loss {average_decoder_loss:.3f}, mask decoder loss {average_masked_decoder_loss:.3f}, uses {used_time:.1f} seconds.'
                    Macro_P = np.mean(list(precisions.values()))
                    Macro_R = np.mean(list(recalls.values()))
                    Macro_F1 = np.mean(list(f1s.values()))
                    for i in range(len(f1s)):
                        logs += f'''\n      Label_{i}: Precision={precisions[i]:.3f},  Recall={recalls[i]:.3f}, F1:{f1s[i]:.3f};'''
                    logs += f'''\n      Macro_P={Macro_P:.3f},  Macro_R={Macro_R:.3f}, Macro_F1={Macro_F1:.3f}.'''
                    logger.logger.info(logs)

                if args.masked_lm:
                    cur_loss = average_encoder_loss + average_masked_decoder_loss
                else:
                    cur_loss = average_encoder_loss + average_decoder_loss
                if cur_loss < best_loss:
                    if args.masked_lm:
                        best_loss = average_encoder_loss + average_masked_decoder_loss
                    else:
                        best_loss = average_encoder_loss + average_decoder_loss
                    if args.local_rank in [-1, 0]:
                        model_to_save = model.module if hasattr(model, "module") else model
                        # Simple serialization for models and tokenizers
                        save_path = '../checkpoints/large-finetune-model'
                        logger.logger.info('保存模型到路径：{}'.format(save_path))
                        model_to_save.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)

                if args.local_rank in [-1, 0]:
                    step_path = f'{save_path}/global_steps{global_steps}'
                    if not os.path.exists(step_path):
                        os.makedirs(step_path)
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(step_path)
                        tokenizer.save_pretrained(step_path)

                scheduler.step(average_decoder_loss)
                start = time.time()
                total_loss = 0
                total_encoder_loss = 0
                total_decoder_loss = 0
                local_step = 0

