import json
from tqdm import tqdm

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# loads BLIP-2 pre-trained model
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)

def main():
    generated_qa_data_path = 'outputs/vq2_qa_pairs.json'
    output_generated_qa_data_vqa_path = 'outputs/vq2_qa_pairs_vqa.json'
    vqa_prompt = 'is "<aj>" true for "<qj>" in this image?'

    generated_qa_data = json.load(open(generated_qa_data_path, 'r'))

    for image_text_data in generated_qa_data:
        print(f"Image: {image_text_data['image']}")
        print(f"Caption: {image_text_data['caption']}")
        raw_image = Image.open(image_text_data['image']).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        for qa_pair in tqdm(image_text_data['qa_pairs'], desc='Generating VQA answers', total=len(image_text_data['qa_pairs'])):
            question, text_answer_candidate = qa_pair['question'], qa_pair['answer']
            qa_pair_prompt = vqa_prompt.replace('<aj>', text_answer_candidate).replace('<qj>', question)
            vqa_answer = model.generate({"image": image, "prompt": f"Question: {qa_pair_prompt} Answer:"})
            qa_pair['vqa_answer'] = vqa_answer[0]['text']

    json.dump(generated_qa_data, open(output_generated_qa_data_vqa_path, 'w'))
    print(f"Saved generated QA data to {output_generated_qa_data_vqa_path}")


if __name__ == '__main__':
    main()
