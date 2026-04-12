# Deep Learning — Wind Turbine Detection & Counting

## Index

1. [Problem Definition](#problem-definition)
2. [Dependencies](#dependencies)
3. [Auxiliary Functions / Code](#auxiliary-functions--code)
4. [Load Data](#load-data)
5. [Explore the Data](#explore-the-data)
6. [Transform Data](#transform-data)
7. [Modeling — Stage 1: Binary Classification (DS1)](#modeling--stage-1-binary-classification-ds1)
8. [Modeling — Stage 2: Turbine Count (DS2)](#modeling--stage-2-turbine-count-ds2)
9. [Modeling — Stage 3: Transfer Learning (DS2)](#modeling--stage-3-transfer-learning-ds2)
10. [Modeling — Stage 4: Return to Binary Classification](#modeling--stage-4-return-to-binary-classification)
11. [Predict for the Test Set](#predict-for-the-test-set)
12. [The Utility of the Models Created](#the-utility-of-the-models-created)
13. [Future Work](#future-work)

---

## Problem Definition

### Context

Wind turbines generate electricity from wind. More and more states and private companies are investing in renewable energies and building wind turbine plants. Knowing the installed capacity and wind turbine locations could help in energy production forecast — in the case of wind turbines, public records are very accurate and the whole fleet is already mapped in databases, but this exercise serves a wider purpose. Deep Learning could help identify and count wind turbines from satellite imagery.

A similar exercise could be done for example on solar panels, for which there is distributed micro generation which is harder to completely map from public databases.

### Objective

Considering the above, the objective is **Turbine Counting**. To start small and guarantee a working model before tackling a harder problem, the objectives are split into two stages:

1. Perform **Binary Classification**, detecting if images contain or do not contain wind turbines.
2. Perform **Turbine Counting** (Multi-class Classification).
   - Turbine counting would make more sense as a regression problem, but since the first stage is classification, this assignment follows with classification to avoid forking the narrative.

The model output will not be the raw number of turbines as in regression, but a bucket class indicating which count range the image belongs to.

### Metric
Accuracy (primary). The target is **95%** for this problem.

- **Stage 1**: Accuracy is acceptable — class imbalance (~57/43) is moderate and consistent across splits.
- **Stage 2**: Accuracy is the . Quadratic Weighted Kappa (QWK) is reported for reference, as the buckets are ordered and class imbalance is significant. QWK > 0.6 is considered substantial agreement.

---

## The Utility of the Models Created

| Model | Description | Stage | Purpose | Techniques Applied | Val Accuracy | Test Accuracy | QWK (Test) |
|---|---|---|---|---|---|---|---|
| model_stg1_baseline | Custom CNN | 1 — DS1 | Turbine detection | Custom CNN, 128×128 | ~95% | 93.00% | — |
| model_stg1_v2 | Custom CNN + L2 + Dropout | 1 — DS1 | Turbine detection | + Dropout, regularisation | ~98.4% | 68.50% | — |
| model_stg1_v3 | Custom CNN + L2 on Conv + dual Dropout | 1 — DS1 | Turbine detection | + L2 on all Conv layers, dual Dropout, early stop | ~98.6% | 93.00% | — |
| model_stg2_softmax | Custom CNN | 2 — DS2 | Count (7 buckets) | Custom CNN, bucketed softmax | 47.16% | 51.46% | 0.31 |
| model_stg2_softmax_cw | Custom CNN | 2 — DS2 | Count (7 buckets) | + Class weights | 38.38% | 40.10% | 0.25 |
| model_stg2_softmax_aug_cw | Custom CNN | 2 — DS2 | Count (7 buckets) | + Augmentation + class weights | 40.45% | 38.73% | 0.04 |
| model_stg3_mobilenet_sft | MobileNetV2 transfer | 3 — DS2 | Count (7 buckets) | MobileNetV2 frozen, 128×128 | 44.92% | 39.07% | 0.41 |
| model_stg3_mobilenet_sft_v2 | MobileNetV2 transfer | 3 — DS2 | Count (7 buckets) | + Smaller head, augmentation | 44.92% | 36.32% | 0.41 |
| model_stg3_effnet_sft | EfficientNetB0 transfer | 3 — DS2 | Count (7 buckets) | EfficientNetB0 frozen, 224×224 | 72.98%* | 67.47% | 0.78 |
| model_stg3_effnet_sft_v2 | EfficientNetB0 transfer | 3 — DS2 | Count (7 buckets) | + L2 reg, dropout, class weights, augmentation | 52.67% | 50.77% | 0.54 |
| model_stg4_effnet_binary_aug | EfficientNetB0 transfer | 4 — DS2 | Turbine detection | EfficientNetB0 frozen + Augmentation | 87.61% | 82.79% | — |

*Severe overfitting — training accuracy ~99%, val loss diverging

---

## Future Work

### Personal Reflection

This assignment was a great learning experience on many aspects: both technical details, but mostly about methodology and how it can be tricky to gauge the amount of work.

One of the learnings was dismissing the depth of the first dataset after getting 95% accuracy with a simple model. By doing so, the project overcomplicated by adding a second stage with a second dataset (knowing it was not a great dataset), when in hindsight there was much more to explore and optimise by working with the first dataset. Even adding the second dataset could have been limited to binary classification — there was no need to explore multiclass counting given the dataset size.

Jumping to conclusions too quickly and committing to a storyline too early was a key lesson. 

### Different Approaches to the Same Problem (Dataset 2)

Improvements still to be explored for the turbine count problem:

- **Unfreeze top backbone layers** so the pretrained model can adapt to satellite imagery (ImageNet features are very different from satellite images).
- **Train at higher resolution** — possibly combined with tiling images into smaller patches, predicting per tile and summing detections.
- **Mix images from Dataset 1**, play with different zoom levels as a form of augmentation.
- **Bounding box detection** — use what the original labels were built for: predict turbine bounding boxes and count the number of boxes per image (YOLO, Faster R-CNN).

### Different Problem Framing

- Tackling the count problem with **regression** instead of bucketed multiclass classification.
- **Density map regression** (crowd counting approach) — predict a 2D density map whose pixel values sum to the turbine count.
- **Ordinal loss** — exploit the ordered nature of count buckets directly in the loss function.

### Looking Further Ahead

- Estimate turbine sizes from imagery.
- Apply the same approach to solar panels and estimate panel area.