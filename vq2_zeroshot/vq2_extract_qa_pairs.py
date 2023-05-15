import json

import matplotlib.pyplot as plt
from PIL import Image

from question_generation import get_answer_candidates, get_questions_sample

def main():
    images_path = ['images/id_22_Colors_sd_2_1_2.png', 'images/COCO_val2014_000000067122.jpg']
    captions = ['A black apple and a green backpack', 'The dog is resting on the chair on the porch of the house.']
    json_out_path = 'outputs/vq2_qa_pairs.json'
    vq2_data = []
    total_qa_pairs = 0

    for image_path, caption in zip(images_path, captions):
        image = Image.open(image_path)
        plt.imshow(image)
        plt.suptitle(caption, fontsize=14)
        plt.show()

        row_vq2_data = {'image': image_path, 'caption': caption, 'qa_pairs': []}
        caption_answer_candidates = get_answer_candidates(caption)
        for answer in caption_answer_candidates:
            questions = get_questions_sample(answer, caption)
            for question in questions:
                row_vq2_data['qa_pairs'].append({'question': question, 'answer': answer})
        vq2_data.append(row_vq2_data)
        total_qa_pairs += len(row_vq2_data['qa_pairs'])

    json.dump(vq2_data, open(json_out_path, 'w'), indent=4)
    print(f"Saving {total_qa_pairs} QA pairs to {json_out_path}")

    print("\nGenerated Question-Answering Pairs:\n")
    for qa_pair in vq2_data:
        print(f"Image: {qa_pair['image']}")
        print(f"Caption: {qa_pair['caption']}")
        print(f"QA Pairs: {qa_pair['qa_pairs']}")
        print()

if __name__ == '__main__':
    main()