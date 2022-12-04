# DGSA

This is the implementation of [Joint Aspect Extraction and Sentiment Analysis with Directional Graph Convolutional Networks](https://www.aclweb.org/anthology/2020.coling-main.24/) at COLING 2020.

You can e-mail Yuanhe Tian at `yhtian@uw.edu`, if you have any questions.

## Citation

If you use or extend our work, please cite our paper at COLING 2020.

```
@inproceedings{chen-etal-2020-joint-aspect,
    title = "Joint Aspect Extraction and Sentiment Analysis with Directional Graph Convolutional Networks",
    author = "Chen, Guimin  and Tian, Yuanhe  and Song, Yan",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    pages = "272--279",
}
```

## Requirements

Our code works with the following environment.
* `python=3.7`
* `pytorch=1.3`

## Dataset

To obtain the data, you can go to [`data`](./data) directory for details.

## Downloading BERT and DGSA

In our paper, we use BERT ([paper](https://www.aclweb.org/anthology/N19-1423/)) as the encoder.

For BERT, please download pre-trained BERT-Base and BERT-Large English from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

For DGSA, you can download the models we trained in our experiments from [Google Drive](https://drive.google.com/drive/folders/1U78sBVGn5Uj0EP-nSl46LFgS8RgbJxJ9?usp=sharing) or [Baidu Net Disk](https://pan.baidu.com/s/1eaY8KBXj3z_gfST7MpNqMw) (passcode: u6gp).

## Run on Sample Data

Run `run_sample.sh` to train a model on the small sample data under the `sample_data` directory.

## Training and Testing

You can find the command lines to train and test models in `run_train.sh` and `run_test.sh`, respectively.

Here are some important parameters:

* `--do_train`: train the model.
* `--do_eval`: test the model.

## To-do List

* Regular maintenance.

You can leave comments in the `Issues` section, if you want us to implement any functions.

