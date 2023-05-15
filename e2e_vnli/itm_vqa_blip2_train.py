import argparse
import os
import random
import subprocess
import tempfile

import accelerate

import numpy as np
import pandas as pd
import torch
import tqdm
from PIL import Image
from lavis.common.registry import registry
from lavis.models import load_preprocess
from omegaconf import OmegaConf

def load_model_with_custom_configs(name, model_type, config='blip2_pretrain_flant5xl_highres.yaml',
                                   is_eval=False, device="cpu", image_size=224, max_length=128):
    model_cls = registry.get_model_class(name)
    cfg = OmegaConf.load(config)
    model = model_cls.from_config(cfg.model)

    model.max_text_length = max_length
    model.max_txt_len = max_length
    preprocess_cfg = cfg.preprocess
    vis_processors, txt_processors = load_preprocess(preprocess_cfg)

    if device == "cpu":
        model = model.float()

    if is_eval:
        model.eval()

    return model.to(device), vis_processors, txt_processors


def load_preprocess_with_custom_configs(config='blip2_pretrain_flant5xl_highres.yaml'):
    cfg = OmegaConf.load(config)
    preprocess_cfg = cfg.preprocess
    vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    return vis_processors, txt_processors


class VQ2R2Dataset(torch.utils.data.Dataset):
    def __init__(self, data, vis_processors, txt_processors, args, training=False):
        self.args = args
        self.data = data
        self.training = training
        self.vis_processor = vis_processors['train'] if training else vis_processors['eval']
        self.txt_processor = txt_processors['train'] if training else txt_processors['eval']


    def __getitem__(self, idx):
        c_data = self.data[idx]
        bdir = args.image_dir if self.training else args.test_image_dir
        image_path = os.path.join(bdir, c_data['image'])
        if not image_path.endswith('.jpg') and not image_path.endswith('.png'):
            image_path = image_path + '.png'
        image = Image.open(image_path)
        image = self.vis_processor(image.convert('RGB'))
        completion = str(c_data['label'])
        completion = 'yes' if completion == '1' else 'no'
        text = c_data['text']
        prompt = f'Does this image entail the description: "{text}"?'
        return {'image':image, 'prompt': prompt, 'completion': completion, 'image_path': image_path, 'dataset_source': c_data['dataset_source']}

    def __len__(self):
        return len(self.data)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train')
    parser.add_argument('val')

    parser.add_argument('--batch_size',
                        type=int,
                        default=2)

    parser.add_argument('--n_epochs',
                        type=int,
                        default=5)

    parser.add_argument('--workers_dataloader',
                        type=int,
                        default=8)

    parser.add_argument(
        '--image_dir',
        default=None,
        required=True)

    parser.add_argument(
        '--model',
        choices=['xl', 'xxl'],
        default='xl',
        type=str
    )

    parser.add_argument(
        '--text_only',
        default=0,
        type=int,
    )

    parser.add_argument('--lr',
                        type=float,
                        default=.00001)

    parser.add_argument('--output_dir',
                        type=str,
                        default='t5_model_outputs_vq2_wide_train')

    parser.add_argument('--use_accelerate',
                        type=int,
                        default=0)

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help='how many steps for gradient accumulation')
    parser.add_argument('--load_from',
                        type=str,
                        default=None,
                        help='which checkpoint to load from?')

    parser.add_argument('--run_name',
                        type=str,
                        default='')

    parser.add_argument('--temp_dir',
                        default='tmpdir'
                        )

    parser.add_argument(
        '--test_image_dir',
        default='all_vq2_test_images')

    args = parser.parse_args()
    args.model_type = 'pretrain_flant5xl' if args.model == 'xl' else 'pretrain_flant5xxl'
    args.config = 'blip2_pretrain_flant5xl_highres.yaml' if args.model == 'xl' else 'blip2_pretrain_flant5xxl_highres.yaml'
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.output_model_path = '{}/{}~batch={}~lr={}'.format(
        args.output_dir, ('run={}~'.format(args.run_name) if args.run_name != '' else '') + args.model_type, args.batch_size, args.lr)

    done = ['{}/'.format(args.output_dir) + x for x in os.listdir(args.output_dir) if '~valloss' in x]
    done_to_match = [x.split('~valloss')[0] for x in done]

    if args.load_from:
        load_from_model_name = args.load_from.split('/')[-1].split(".pt")[0]
        print(f"*** Loading from! {args.load_from}")
        print(f"load_from_model_name: {load_from_model_name}")
        args.output_model_path += f"_load_from_{load_from_model_name}"

    if args.output_model_path in done_to_match:
        print('{} already done!'.format(args.output_model_path))
        quit()

    ''' add number of epochs to model name '''
    args.output_model_path += '~n_epochs={}'.format(args.n_epochs)
    args.output_model_path += '~valloss={:.4f}.pt'

    print(args.output_model_path)

    return args


def main():
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    accelerator = accelerate.Accelerator()
    mainproc = accelerator.is_local_main_process

    # setup device to use
    args.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model, vis_processors, text_processors = load_model_with_custom_configs(
        name="blip2_t5", config=args.config, model_type=args.model_type, is_eval=False,
        device=args.device, image_size=364, max_length=128
    )
    model.bfloat16()
    if args.load_from:

        state = torch.load(args.load_from, map_location=torch.device('cpu'))
        state['model_state_dict'] = {k.replace('module.', '') : v for k, v in state['model_state_dict'].items()}
        model.load_state_dict(state['model_state_dict'])

    val = pd.read_csv(args.val).to_dict('records')
    val = torch.utils.data.DataLoader(
        VQ2R2Dataset(val, vis_processors, text_processors, args, training=False),
        batch_size=args.batch_size, num_workers=args.workers_dataloader, shuffle=False, worker_init_fn=worker_init_fn)

    train = pd.read_csv(args.train).to_dict('records')
    train = torch.utils.data.DataLoader(
        VQ2R2Dataset(train, vis_processors, text_processors, args, training=True),
        batch_size=args.batch_size, num_workers=args.workers_dataloader, shuffle=True, worker_init_fn=worker_init_fn)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    if args.use_accelerate:
        model, optim, train = accelerator.prepare(model, optim, train)

    best_val_loss = np.inf
    not_improved_epoch = 0

    if mainproc:
        tmpfile = tempfile.NamedTemporaryFile()
        print('using tempfile {}'.format(tmpfile.name))

    for epoch in range(args.n_epochs):
        print('Epoch {}'.format(epoch))
        for mode in ['train', 'val']:
            if mode == 'train':
                model.train()
                bar = tqdm.tqdm(enumerate(train), total=len(train), disable=not mainproc)
            else:
                model.eval()
                bar = tqdm.tqdm(enumerate(val), total=len(val), disable=not mainproc)
            n, running_sum_loss = 0, 0
            with torch.set_grad_enabled(mode=='train'):
                for i, b in bar:
                    batch = {}
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        batch['text_input'] = [b['prompt'][idx] for idx in range(len(b['image']))]
                        batch['text_output'] = [b['completion'][idx] for idx in range(len(b['image']))]
                        batch['image'] = b['image'].to(args.device).bfloat16()
                        loss = model(batch)['loss'].mean()
                    running_sum_loss += loss.cpu().detach().numpy()
                    n += 1
                    if mode == 'train':
                        loss_scaled = loss / args.gradient_accumulation_steps
                        if args.use_accelerate:
                            accelerator.backward(loss_scaled)
                        else:
                            loss_scaled.backward()

                        if i % args.gradient_accumulation_steps == 0 or i == len(train) - 1:
                            optim.step()
                            optim.zero_grad()
                    bar.set_description('loss = {:.6f}'.format(running_sum_loss / n))

                if mode == 'val' and mainproc:
                    val_loss = running_sum_loss / n
                    best_yet = val_loss < best_val_loss
                    if best_yet:
                        best_val_loss = val_loss

                        if args.use_accelerate:
                            torch.save(
                                {'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                                 'args': vars(args)},
                                tmpfile.name)
                        else:
                            try:
                                torch.save(
                                    {'model_state_dict': model.module.state_dict(),
                                     'args': vars(args)},
                                    tmpfile.name)
                            except:
                                torch.save(
                                    {'model_state_dict': model.state_dict(),
                                     'args': vars(args)},
                                    tmpfile.name)
                        not_improved_epoch = 0
                    else:
                        not_improved_epoch += 1

    accelerator.wait_for_everyone()
    if mainproc:
        args.output_model_path = args.output_model_path.format(best_val_loss)
        subprocess.call('cp {} {}'.format(tmpfile.name, args.output_model_path), shell=True)

if __name__ == '__main__':
    args = parse_args()
    main()
