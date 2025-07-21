# Data preparation:

Prepare the VG dataset and point the /path/to/image in the script to the VG dataset.
The full dataset can be found at [MM-GCoT](https://huggingface.co/datasets/AQUA6/MM-GCoT/viewer/train?views%5B%5D=train&row=8).

# Evaluation:

Execute evaluation scripts through the provided interface in ./scripts directory:


- "SPLIT" refers to the dataset type, including "attributes", "judge", and "objects".
- "CKPT" refers to the model name.
- "answer_first" determines the generation order: 
    - 1, the MLLM first generates the answer and then the bbox; 
    - 0, the MLLM first generates the bbox and then the answer.

