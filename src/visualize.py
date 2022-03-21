from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model
from logger import Logger
from datasets.dataset_factory import get_dataset
from extra.extractor import PclExtractor

def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    # opt.dataset: default is 'coco', opt.task 'ctdet'
    ## 1.prepare dataset
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    dataset = Dataset(opt, 'train')
    print(opt)
    logger = Logger(opt)

    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    ## 2. prepare model
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model.to(opt.device)


    if opt.load_model != '':
        model = load_model(model, opt.load_model)

    ## 3. prepare extractor

    extractor = PclExtractor(opt, model, dataset)

    print('Starting extracting...')
    extractor.run()
    print('End of extracting...')

    logger.close()

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)