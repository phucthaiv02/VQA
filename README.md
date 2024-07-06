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
