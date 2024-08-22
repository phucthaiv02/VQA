# VQA - Visual question answering

## Usage
### Prerequisites
- Python 3.9
- Pytorch 2.3 with CUDA 11.8

### Data setup
- Download data via this [link](https://drive.google.com/file/d/1P2AkW26lUjy6QO8_DsJVlifMX0Q4IdPA/view?usp=sharing) and extract to folder ```data/```.
- Folder ```data/``` looks like this: 
```
VQA/
    ├── data/
    │       ├── raw/
    │       │     ├── images/
    │       │     │   ├── image1.png
    │       │     │   ├── image2.png
    │       │     │   └── ...
    │       │     ├── all_qa_pairs.txt
    │       │     ├── answer_space.txt
    │       │     ├── data.csv
    │       │     └── ...
    │       ├── datasets.py
    │       └── ...
```
This dataset contains the processed DAQUAR Dataset (full), along with some of the raw files from the original dataset.

Details:
- **data.csv**: This is the processed dataset after normalizing all the questions & converting the *{question, answer, image_id}* data into a tabular format for easier consumption.
- **data_train.csv**: This contains those records from data.csv which correspond to images present in **train_images_list.txt**
- **data_eval.csv**: This contains those records from data.csv which correspond to images present in **test_images_list.txt**
- **answer_space.txt**: This file contains a list of all possible answers extracted from **all_qa_pairs.txt** (This will allow the VQA task to be modelled as a multi-class classification problem)


### Training 
To train model, run the following command:
```
    python main.py 
        --train 
        --train_data [path to training data <data_train.csv>]
        --eval_data [path to evaluation data <data_eval.csv>] 
        --image_dir [path to image folder]
        --output [path to save model checkpoint]
```

### Inference
*[Temporarily not in use]*

To test your model on real data, run the following command:
```
    python main.py
        --infer
        --image [path to your image]
        --question [question about your image]
        --model [path to your model]
```