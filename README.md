# [SIOD: Single Instance Annotated Per Category Per Image for Object Detection](https://arxiv.org/abs/2203.15353)

## Main Results 

| Detector                                 | Task         | $AP@\mathbb{S}$ | $AP@\mathbb{S}_0$ | $AP@\mathbb{S}_3$ | $AP@\mathbb{S}_5$ | $AP@\mathbb{S}_7$ | $AP@\mathbb{S}_9$ |
| ---------------------------------------- | ------------ | --------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| [CenterNet-Res18](https://drive.google.com/drive/folders/1VDpT8J9SWq-KV7_XH2YQ6Pflsgprgs2r) | FSOD         | 17.3            | 28.1              | 24.0              | 17.1              | 8.8               | 1.5               |
| [CenterNet-Res18](https://drive.google.com/drive/folders/1u_1QtMMa-TbjRMVyWsAlw31hzXVbuvho) | SIOD(base)   | 13.9            | 25.1              | 18.5              | 12.3              | 6.1               | 1.4               |
| [CetnerNet-Res18](https://drive.google.com/drive/folders/1N5TBdp1aMDYG1r6AK4WFtyADoe_WAIPd) | SIOD(DMiner) | 16.8(+2.9)      | 26.6(+1.5)        | 22.4(+3.9)        | 17.1(+4.8)        | 9.4(+3.3)         | 2.1(+0.7)         |
| [CenterNet-Res101](https://drive.google.com/drive/folders/1wBl_oh3PRFjSz0Mfsi0jxp-pd6jh6J0_) | FSOD         | 22.6            | 34.2              | 30.3              | 23.6              | 13.6              | 3.1               |
| [CenterNet-Res101](https://drive.google.com/drive/folders/1EOrUl91PlkIPz-VmW9GmuECvT3f22w6p) | SIOD(base)   | 15.1            | 27.8              | 20.9              | 13.3              | 6.1               | 1.1               |
| [CenterNet-Res101](https://drive.google.com/drive/folders/12L17g3haHku2JWfuQNfcm16e6U6hRONZ) | SIOD(DMiner) | 19.7(+4.6)      | 29.8(+2.0)        | 26.0(+5.1)        | 20.5(+7.2)        | 12.2(+6.1)        | 2.9(+1.8)         |

## [Dataset](https://drive.google.com/drive/folders/1mJayvvNkmvur7IOG17-hz3AHQ2yPWfUf) 

- Keep1_COCO2017_Train: **keep1_instances_train2017.json**

- Semi-supervised annotation which has equivalent instance annotations to Keep1_COCO2017_Train: **mark_semi_instances_train2017.json**. (For Table 2)

  We add a new field("keep") to the image infomation in annotation file, where keep=1 indicates the image belongs to labeled part and keep=0 indicates the image belongs to  unlabeled part.

## Preparations

```
1. pip install -r requirements.txt 
2. install pytorch=1.7.0(higher version has some problems in following installation of dcnv2) 
3. install dcnv2
   cd src/lib/models/networks/DCNv2
   sh make.sh 
4. install cocoapi
   cd src/lib/datasets/dataset/cocoapi/
   sh install.sh 
5. install nms
   cd src/lib/external
   make 
6. create soft link for the data
   vim link.sh
   sh link.sh 
```

## Training 

Take CenterNet-Res18 for example:

- Directly train the centernet under SIOD setup.

  ```shell
  sh base_resdcn18_train.sh
  ```

- Train the centernet equipped with SPLG or PGCL.

  ```
  # SPLG
  sh plg_resdcn18_train.sh
  # PGCL 
  gcl_resdcn18_train.sh
  ```

- Train the centernet equipped with DMiner.

  ```
  dminer_resdcn18_train.sh
  or 
  all_resdcn18_train.sh
  ```

## Evaluation 

Evaluate the detector with new Score-aware Detection Evaluation Protocol.

```
# modify the parameter "load_model" accordingly
sh test_resdcn18.sh
```

## Visualization 

Prepare some images and modified visualize.sh accordingly. 

```
sh visualize.sh
```

