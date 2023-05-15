
import argparse
import random

import itm_vqa_blip2_train

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from transformers import T5TokenizerFast

images_path = 'all_vq2_test_images'
data_path = 'final_test_set.csv'
vq2_test_res_dir = 'vq2_test_res'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_set')

    parser.add_argument('load_from',
                        type=str,
                        help='which checkpoint to load from?')

    parser.add_argument('--workers_dataloader',
                        type=int,
                        default=8)

    parser.add_argument(
        '--image_dir',
        default='all_vq2_test_images')

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

    args = parser.parse_args()
    model_load_from_directory = args.load_from.split('/')[0]
    print(f"model_load_from_directory: {model_load_from_directory}")
    args.model_type = 'pretrain_flant5xl' if args.model == 'xl' else 'pretrain_flant5xxl'
    extra_name = ''
    args.config = 'blip2_pretrain_flant5xl_highres.yaml' if args.model == 'xl' else 'blip2_pretrain_flant5xxl_highres.yaml'

    print(f"Loading from: {args.load_from}")

    if 'load_from' in args.load_from:
        extra_name += f"_load_from_previous_model"

    print(f"extra_name: {extra_name}")
    args.output_path = '{}/vq2_round2_predsby={}~dataset={}'.format(model_load_from_directory, args.model_type + extra_name,
                                                           args.test_set.split('/')[-1])
    print(f"args.output_path: {args.output_path}")
    args.batch_size = 1

    return args


def main():
    args = parse_args()

    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # setup device to use
    args.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model, vis_processors, text_processors = itm_vqa_blip2_train.load_model_with_custom_configs(
        name="blip2_t5", config=args.config, model_type=args.model_type, is_eval=True,
        device=args.device, image_size=364, max_length=128
    )
    model.bfloat16()

    LOAD_MODEL = True
    if LOAD_MODEL:
        print(f"Loading Model")
        state = torch.load(args.load_from, map_location=torch.device('cpu'))
        state['model_state_dict'] = {k.replace('module.', ''): v for k, v in state['model_state_dict'].items()}
        model.load_state_dict(state['model_state_dict'])
    else:
        print(f"This is now the UNLOADED model!!!!!! ")
    model.eval()

    t5_tokenizer = T5TokenizerFast.from_pretrained('google/flan-t5-xl')

    test = pd.read_csv(args.test_set).to_dict('records')
    test = torch.utils.data.DataLoader(
        itm_vqa_blip2_train.VQ2R2Dataset(test, vis_processors, text_processors, args, training=False),
        batch_size=args.batch_size, num_workers=args.workers_dataloader, shuffle=False, worker_init_fn=itm_vqa_blip2_train.worker_init_fn)

    bar = tqdm.tqdm(enumerate(test), total=len(test))
    n, running_sum_loss = 0, 0

    all_predictions = []
    labels = []
    preds = []
    with torch.no_grad():
        for i, b in bar:
            batch = {}
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                batch['text_input'] = [b['prompt'][idx] for idx in range(len(b['image']))]
                batch['text_output'] = [b['completion'][idx] for idx in range(len(b['image']))]
                batch['image'] = b['image'].to(args.device).bfloat16()
                batch['dataset_source'] = [b['dataset_source'][idx] for idx in range(len(b['dataset_source']))]

                loss = model(batch)['loss'].mean()

                res = model.predict_answers(batch, temperature=0.0001)
                raw_pred = res[0]
                pred = 1 if raw_pred == 'yes' else 0
                label = batch['text_output'][0]
                dataset_source = batch['dataset_source'][0]
                labels.append(label)
                preds.append(pred)
                row_pred = {'image_path': b['image_path'][0],
                     'text': b['prompt'][0],
                     'label': b['completion'][0],
                     'prediction': res[0],
                            'raw_pred': raw_pred,
                            'dataset_source': dataset_source}
                all_predictions.append(row_pred)

            running_sum_loss += loss.cpu().detach().numpy()
            n += 1
            bar.set_description('loss = {:.6f}'.format(running_sum_loss / n))

    all_predictions_df = pd.DataFrame(all_predictions)
    all_predictions_df.to_csv(args.output_path, index=False)
    print(f'Wrote predictions to {args.output_path}')
    all_predictions_df['label_int'] = all_predictions_df['label'].apply(lambda x: 1 if x == 'yes' else 0)
    all_predictions_df['prediction_int'] = all_predictions_df['prediction'].apply(lambda x: 1 if x == 'yes' else 0)
    pred_col = 'prediction_int'
    label_col = 'label_int'
    ''' Calc ROC_AUC score per dataset_source group '''
    stats = []
    for dataset_source, df_dataset in all_predictions_df.groupby('dataset_source'):
        num_samples = len(df_dataset)
        num_pos = df_dataset[label_col].sum()
        num_neg = num_samples - num_pos
        roc_auc = roc_auc_score(df_dataset[label_col], df_dataset[pred_col])
        stats.append([dataset_source, num_samples, num_pos, num_neg, roc_auc])
    df_stats = pd.DataFrame(stats, columns=['dataset_source', 'num_samples', 'num_pos', 'num_neg', 'roc_auc'])
    print(df_stats)


if __name__ == '__main__':
    main()