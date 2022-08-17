"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json, sys
import stanza
import spacy_stanza
import numpy as np
import torch as th
from transformers import set_seed
import torch.distributed as dist
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
from improved_diffusion.test_util import get_weights, denoised_fn_round
from functools import partial
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
sys.path.insert(0, 'Diffusion-LM/transformers/examples/pytorch/language-modeling')
from infill_util import langevin_fn3, get_score, langevin_fn3_compose, langevin_fn1, langevin_fn4, langevin_fn_tree, langevin_fn_length
from spacy.lang.en import English

# import debugpy
# debugpy.listen(5678)
# print("waiting for debugger")
# debugpy.wait_for_client()
# print("debugger attached")

def main():
    set_seed(101)
    args = create_argparser().parse_args()
    model_kwargs = {}
    # print all args
    for k, v in vars(args).items():
        print(f'args.{k} = {v}')

    # load configurations.
    print(f"os.path.split(args.model_path)[0]: {os.path.split(args.model_path)[0]}")
    print(f"args.model_path: {args.model_path}")
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    # config_path = f'{os.getcwd()}/diffusion_models/diff_roc-free_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_xstart_e2e/training_args.json'
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        print('loading config')
        training_args = json.load(f)
    args.__dict__.update(training_args)

    args.noise_level = 0.0
    args.sigma_small = True

    args.diffusion_steps = 200  # 500  # DEBUG
    dist_util.setup_dist()
    logger.configure()
    print(args.clip_denoised, 'clip_denoised')
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(f'{os.getcwd()}/{args.model_path}', map_location=dist_util.dev()))
    model.to(dist_util.dev())
    model.eval()

    logger.log("load embedding models")
    print(os.path.split(args.model_path)[0])
    model_embs, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                   os.path.split(args.model_path)[0])
    if args.training_mode.startswith('e2e'):
        print('e2e, load the right model embeddings', '*'*80)
        model_embs.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_embs = model_embs.to(dist_util.dev())
    model3 = get_weights(model_embs, args)
    logger.log('load the partial sequences')
    if args.partial_seq:
        partial_seq = [args.partial_seq]
        partial_seq_idx = ['0']
    elif args.partial_seq_file:
        # implies that we should read from the files
        nlp = English()
        tokenizer_spacy = nlp.tokenizer
        print(f'reading from the file {args.partial_seq_file}', '-*'*20)
        with open(args.partial_seq_file, 'r') as f:
            sent_lst = json.load(f)
        partial_seq = []
        partial_seq_idx = []
        for idx, (key, val) in enumerate(sent_lst.items()):
            if idx < int(args.start_idx) or idx > int(args.end_idx):
                continue
            partial_seq_ = f"{val['obs1']} " + "PAD " * 10 + f"{val['obs2']}"
            word_lst = [x.text for x in tokenizer_spacy(partial_seq_)]
            partial_seq_ = " ".join(word_lst)
            print(partial_seq_, idx)
            partial_seq.append(partial_seq_)
            partial_seq_idx.append(str(idx))
    else:
        partial_seq = ['A kid friendly venue named Alimentum is located on the riverside .',
                       'Alimentum , situated by the river , is quite child friendly .']
        partial_seq_idx = ['0', '1']
    # else:  generate them by randomly preturbing the inputs data.
    if args.modality in ['synth', 'pos']:
        tokens2id = {v:k for k, v in tokenizer.items()}
        todo_pad_token = tokens2id['END']
        print(f'pad token = {todo_pad_token}')
        encoded_partial_seq = [th.LongTensor([tokens2id[x] for x in seq.split()]) for seq in partial_seq]
        print(encoded_partial_seq[0], len(encoded_partial_seq[0]))
    elif args.modality in ['e2e-tgt', 'roc', 'roc-aug','roc-free']:
        tokens2id = {v:k for k, v in tokenizer.items()}
        todo_pad_token = -1
        pad_token = tokens2id['PAD']
        encoded_partial_seq = [th.LongTensor([tokens2id.get(x, tokens2id['UNK']) for x in seq.split()]) for seq in partial_seq]

        if args.eval_task_ == 'free_emotion':
            control_constraints = []
            path_to_rand_emos = f'{os.getcwd()}/datasets/ROCstory/target_emotion_scores_1024.json'
            rand_emotionlist = []
            with open(path_to_rand_emos, 'r') as rr:
                for emotion in rr:
                    emotion = json.loads(emotion)
                    rand_emotionlist.append(emotion)
            emotion_vals = th.tensor(rand_emotionlist)
            # #! take the first emotion and broadcast it to the rest of the emotions.
            # emotion_vals = emotion_vals[0].unsqueeze(0).repeat(emotion_vals.shape[0], 1)
            # print(f"emotion_vals: {emotion_vals}")

            print('in the emotion inference')
            emotion_vals = emotion_vals.split(int(args.batch_size/2))

            encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
            encoded_seq = encoded_partial_seq
            assert len(partial_seq) == len(encoded_partial_seq)
            print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)
            langevin_fn_selected = None


    logger.log("sampling...")
    sample_dict = {}
    for idx, emotion_val in enumerate(emotion_vals):
        all_images = []
        all_labels = []
        while len(all_images) * args.batch_size < args.num_samples:
            encoded_seq = encoded_seq[0].unsqueeze(0).expand(args.batch_size,-1)
            # print(f"encoded_seq.device: {encoded_seq.device}")
            # print(f"model_embs.weight.device: {model_embs.weight.device}")
            partial_mask_temp = (encoded_seq == todo_pad_token).view(args.batch_size, -1)
            # encoded_seq[encoded_seq == todo_pad_token] = 0
            encoded_seq.masked_fill_(encoded_seq == todo_pad_token, 3)
            # emotion_val_pad = emotion_val.pad(th.zeros_like(emotion_val))
            emotion_val = th.cat((emotion_val, th.zeros_like(emotion_val)), dim=0)
            # pad emotion_val with 
            # encoded_seq_hidden = model_embs(encoded_seq.to(dist_util.dev()))
            seqlen = encoded_seq.size(0)
            partial_mask = partial_mask_temp.unsqueeze(-1).expand(-1, -1, args.in_channel)
            sample_shape = (args.batch_size, seqlen, args.in_channel, )
            model_kwargs = {'emotion': emotion_val}

            if args.eval_task_ == 'free_emotion':
                if args.use_ddim:
                    loop_func_ = diffusion.ddim_sample_loop_progressive
                else:
                    loop_func_ = diffusion.p_sample_loop_progressive
                for sample in loop_func_(
                        model,
                        sample_shape,
                        denoised_fn=partial(denoised_fn_round, args, model3.to(dist_util.dev())),
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                        device=dist_util.dev(),
                        langevin_fn=langevin_fn_selected,
                        eta=args.eta,
                ):
                    final = sample["sample"]


                if args.verbose == 'yes':
                    with open(f'debug_lst_lgv_{args.notes}.json', 'w') as f:
                        json.dump(debug_lst, f)
                    label_ids = label_ids.expand(args.batch_size, -1).to(dist_util.dev())
                    tgt_embs = model3(label_ids[:, final.size(1):])

                    label_ids2 = th.cat([label_ids[:, :final.size(1)], label_ids], dim=1)
                    label_ids2[:, :64 * 2 + 1] = -100
                    tt = th.LongTensor([0]).expand(final.size(0)).to(final.device)
                    prev_sample = diffusion.q_sample(final, tt)
                    input_embs = th.cat([final, prev_sample, tgt_embs], dim=1)
                    model_out = model_control(input_embs=input_embs,
                                                labels=label_ids2)
                    print(model_out.loss, 'final end')
                    loss_fn = th.nn.CrossEntropyLoss(reduction='none')
                    shifted_logits = model_out.logits[:, :-1].contiguous()
                    shifted_labels = label_ids2[:, 1:].contiguous()
                    loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)),
                                    shifted_labels.view(-1)).reshape(shifted_labels.shape)
                    print(loss.sum(dim=-1).tolist())
                    word_lst = rounding_func(args.experiment, final, model3, tokenizer)
                    print(len(word_lst))
                    for ww, ll in zip(word_lst, loss.sum(dim=-1).tolist()):
                        print([ww], ll)


            else:
                label_class_attributes = control_helper
                loop_func_ = diffusion.p_sample_loop_progressive_infill


                for sample in loop_func_(
                        model,
                        sample_shape,
                        encoded_seq_hidden,
                        partial_mask,
                        denoised_fn=partial(denoised_fn_round, args, model3.to(dist_util.dev())),
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                        device=encoded_seq_hidden.device,
                        greedy=False,
                ):
                    final = sample["sample"]

            sample = final

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            if args.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images) * args.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        arr2 = arr[:int(args.batch_size / 2)]
        arr = arr[: args.num_samples]
        if args.verbose == 'pipe':
            if args.eval_task_ == 'free_emotion':
                start_count = int(len(sample_dict)*args.batch_size/2)
                end_count = int(args.batch_size/2)+start_count
                end_count = list(range(start_count, end_count))
                sample_dict[tuple(end_count)] = arr2
                # get length of sample_dict
                print(f"len(sample_dict): {len(sample_dict)}")
                print(
                    f'writing to sample_dict, for {len(sample_dict)*args.batch_size/2}samples')
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'

    dist.barrier()
    logger.log("sampling complete")

    def decode_helper(args, sample_dict, diff_model=None):
        result_dict = {}
        if not diffusion.training_mode.startswith('e2e'):
            logger.log('decode by rounding. ')
            set_seed(101)
            model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                           os.path.split(args.model_path)[0])

        for k, v in sample_dict.items():
            arr = v
            if diffusion.training_mode.startswith('e2e'):
                word_lst_e2e = []
                print('decoding for e2e', )
                x_t = th.tensor(arr).to(dist_util.dev())
                print(x_t.shape)
                if args.model_arch == 'conv-unet':
                    reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
                else:
                    reshaped_x_t = x_t
                logits = diff_model.get_logits(reshaped_x_t) 
                cands = th.topk(logits, k=1, dim=-1)
                tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
                for seq in cands.indices:
                    tokens = " ".join([tokenizer[x[0].item()] for x in seq])
                    word_lst_e2e.append(tokens)
                word_lst = word_lst_e2e
            else:
                word_lst = rounding_func(args.experiment, arr, model, tokenizer)
            result_dict[k] = word_lst
        return result_dict

    if args.verbose == 'pipe':
        print(f'sampled for {len(sample_dict)} control tasks')
        out_path_pipe = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{args.notes}.json")
        result_dict = decode_helper(args, sample_dict, diff_model=model)
        fout = open(out_path_pipe, 'w')
        with open(out_path_pipe, 'w') as out_writer:
            for _, model_output in result_dict.items():
                for i in range(len(model_output)):
                    json.dump([model_output[i]], out_writer)
                    out_writer.write('\n')
        print(f'written the decoded output to {out_path_pipe}')
        out_path2 = out_path_pipe


    elif args.verbose == 'yes':

        if diffusion.training_mode.startswith('e2e'):
            word_lst_e2e = []
            print('decoding for e2e', )
            print(sample.shape)
            x_t = sample
            if args.model_arch == 'conv-unet':
                reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
            else:
                reshaped_x_t = x_t
            logits = model.get_logits(reshaped_x_t)
            cands = th.topk(logits, k=1, dim=-1)
            tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
            for seq in cands.indices:
                tokens = " ".join([tokenizer[x[0].item()] for x in seq])
                word_lst_e2e.append(tokens)
            word_lst = word_lst_e2e
        else:
            logger.log('decode by rounding. ')
            print('load_models')
            set_seed(101)
            print(os.path.split(args.model_path)[0])
            model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel, os.path.split(args.model_path)[0])
            print('rounding')
            word_lst = rounding_func(args.experiment, arr, model, tokenizer)

        out_path2 = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{shape_str}_{args.notes}.txt")
        fout = open(out_path2, 'w')
        for (xx) in zip( word_lst):
            print(xx[0], file=fout)
        fout.close()
        print(f'written the decoded output to {out_path2}')

        ##############
        out_path2 = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{shape_str}_{args.notes}.json")
        fout = open(out_path2, 'w')
        for (xx) in zip(word_lst):
            print(json.dumps(xx), file=fout)
        fout.close()
        print(f'written the decoded output to {out_path2}')


    args.out_path2 = out_path2
    return args

def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=False, use_ddim=False, eta=1.0, num_samples=50, batch_size=1, model_path="",
        out_dir="out_gen",
        emb_scale_factor=1.0, split='train', debug_path='', eval_task_='infill',
        partial_seq="", partial_seq_file="", verbose='yes', tgt_len=15, t_merge=200, interp_coef=0.5, notes='',
        start_idx=0, end_idx=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def eval(args):
    if args.modality == 'e2e-tgt':
        model_name_path = "predictability/diff_models/e2e-tgt_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_None"

        COMMAND = f"python scripts/ppl_under_ar.py " \
              f"--model_path {args.model_path} " \
              f"--modality {args.modality}  --experiment random " \
              f"--model_name_or_path {model_name_path} " \
              f"--input_text {args.out_path2}  --mode eval"
        print(COMMAND)
        os.system(COMMAND)


if __name__ == "__main__":
    args = main()
    import numpy as np
    if args.verbose != 'pipe':
        eval(args)

