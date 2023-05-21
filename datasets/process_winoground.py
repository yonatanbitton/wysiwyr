import pandas as pd
from copy import deepcopy
winoground_path = 'examples.jsonl'
winoground_output_path = 'winoground_cartesian.csv'

def main():
    df = pd.read_json(winoground_path, lines=True)
    items = []
    for img in ['image_0', 'image_1']:
        for caption in ['caption_0', 'caption_1']:
            img_caption = img + "_" + caption
            for _, r in df.iterrows():
                r_dict = deepcopy(r.to_dict())
                r_dict['img_caption'] = img_caption
                r_dict['image'] = r[img]
                r_dict['caption'] = r[caption]
                items.append(r_dict)
    out_df = pd.DataFrame(items)
    out_df.to_csv(winoground_output_path, index=False)
    print("Done")

if __name__ == '__main__':
    main()