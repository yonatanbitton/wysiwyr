import json
import matplotlib.pyplot as plt
from PIL import Image

generated_qa_data_vqa_path = 'outputs/vq2_qa_pairs_vqa.json'


def main():
    generated_vqa_data = json.load(open(generated_qa_data_vqa_path, 'r'))
    for image_text_vqa_data in generated_vqa_data:
        print(f"Image: {image_text_vqa_data['image']}")
        print(f"Caption: {image_text_vqa_data['caption']}")
        image = Image.open(image_text_vqa_data['image']).convert("RGB")

        # Set up the grid
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        # Display the image and caption
        ax1.imshow(image)
        ax1.set_title(image_text_vqa_data['caption'])
        ax1.axis('off')

        # Display the QA pairs
        qa_text_col_2 = ""
        qa_text_col_3 = ""
        for i, vqa_pair in enumerate(image_text_vqa_data['qa_pairs']):
            question, text_answer_candidate, vqa_answer = vqa_pair['question'], vqa_pair['answer'], vqa_pair[
                'vqa_answer']
            qa_text = f"{i + 1}. Q: {question}\n   A: {text_answer_candidate}\n   VQA: {vqa_answer}\n\n"

            # Split QA pairs between the two columns
            if i % 2 == 0:
                qa_text_col_2 += qa_text
            else:
                qa_text_col_3 += qa_text

        ax2.text(0, 1, qa_text_col_2, fontsize=12, ha='left', va='top', wrap=True, transform=ax2.transAxes)
        ax3.text(0, 1, qa_text_col_3, fontsize=12, ha='left', va='top', wrap=True, transform=ax3.transAxes)

        ax2.axis('off')
        ax3.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
