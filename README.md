# Hand Gesture Recognition — CNN

**Task 04 — Machine Learning Internship @ Prodigy InfoTech**

A convolutional neural network that classifies static hand gesture images into 10 categories, enabling gesture-based human-computer interaction.

## Dataset

[LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) — near-infrared hand gesture images captured with a Leap Motion sensor.

```
leapgestrecog/
    00/                     (subject 0)
        01_palm/
        02_l/
        03_fist/
        04_fist_moved/
        05_thumb/
        06_index/
        07_ok/
        08_palm_moved/
        09_c/
        10_down/
    01/                     (subject 1)
        ...
    ...
    09/                     (subject 9)
```

- 10 subjects, each performing 10 gestures
- ~200 images per gesture per subject → ~20,000 grayscale images total
- Gestures: palm, l, fist, fist_moved, thumb, index, ok, palm_moved, c, down

## Approach

Unlike the earlier classical-ML tasks (linear regression, K-means, HOG+SVM), this task uses a **CNN**, since gesture recognition depends on learned spatial/shape features that hand-crafted descriptors capture less effectively across 10 fine-grained classes.

## Files

- `gesture_recognition_cnn.py` — full pipeline: image loading, CNN architecture, training, evaluation, and model export
- `training_curves.png` — accuracy/loss curves across epochs
- `confusion_matrix.png` — confusion matrix on the held-out test set
- `gesture_cnn_model.keras` — saved trained model, with a `predict_gesture()` helper for classifying new images

## Model Architecture

3 convolutional blocks (Conv2D → BatchNorm → MaxPool, filter sizes 32/64/128) → Flatten → Dense(128) → Dropout(0.5) → Softmax(10). Trained with Adam, early stopping on validation accuracy, and learning-rate reduction on plateau.

## ⚠️ Important Caveat: Data Leakage in the Initial Run

The first run of this pipeline used a **random stratified split by image** (70% train / 15% val / 15% test) and reported a suspicious **100% test accuracy** across every class.

This is a known pitfall with LeapGestRecog: each gesture's ~200 images are consecutive frames from a short video of the same pose, so adjacent frames are near-duplicates (same hand, angle, lighting, background). A random image-level split leaks near-identical frames from the same recording into both train and test sets — the model ends up partially memorizing specific frames rather than learning generalizable gesture shapes, which inflates accuracy to a meaningless 100%.

**Correct evaluation requires a subject-independent split** — e.g., training on a subset of subjects (like 00–07) and testing on entirely unseen subjects (like 08–09) — so the model is forced to generalize to hands, lighting, and backgrounds it hasn't seen before. This gives a realistic accuracy estimate instead of an inflated one.

## Results

| Split strategy | Test Accuracy |
|---|---|
| Random image-level split (leaky) | 100% — not a valid estimate |
| Subject-independent split | *To be re-evaluated* |

## Limitations & Next Steps

- Re-run training with a subject-wise train/test split to get a trustworthy accuracy figure.
- Consider data augmentation (rotation, slight scaling, brightness jitter) to improve generalization to new users' hand shapes and lighting conditions.
- For real-time gesture control, the model would also need to handle a "no gesture" / background class and be evaluated on video frames, not just static Leap Motion captures.
- Exploring transfer learning (e.g., a lightweight pretrained CNN) could improve robustness if extending beyond this dataset's controlled near-infrared conditions.

## Requirements

```
numpy
opencv-python
tensorflow
scikit-learn
matplotlib
```

## Usage

Train the model:
```bash
python3 gesture_recognition_cnn.py
```

Classify a new gesture image after training:
```python
from gesture_recognition_cnn import predict_gesture
gesture, confidence = predict_gesture("path/to/new_image.png")
print(gesture, confidence)
```
