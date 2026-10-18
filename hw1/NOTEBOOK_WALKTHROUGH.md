# HW1 — Notebook Walkthrough (Fashion-MNIST)

Notebooks covered: `fashion_pt_1.ipynb`, `fashion_pt_2.ipynb`

**Goal in the top-level README:** representation learning and clustering on Fashion-MNIST: neural-network features, K-means in feature space, failure-case visualisation, and a comparison of raw pixels vs learned embeddings.

> **How the notebooks line up with that goal.** Both notebooks work on Fashion-MNIST with a neural network (an MLP) and use K-means. But K-means is run **only on raw pixels**, and the MLP is used as a **classifier**, not as an embedding extractor. The notebooks never cluster in a learned feature space or compare pixels against embeddings. The README bullets are broader than what the code does. The notebooks' real strength is in debugging a classifier under distribution shift.

---

## Libraries used

| Library | Used for |
|---|---|
| `numpy`, `pandas` | image arrays, DataFrame of `image` / `label` / predictions |
| `torchvision.datasets.FashionMNIST` | loading the 60k training images (only the loader, no PyTorch modelling) |
| `scikit-learn` | `KMeans`, `MLPClassifier`, `MLPRegressor`, `LinearRegression`, `StandardScaler`, `train_test_split`, `confusion_matrix`, `mean_squared_error` |
| `plotly.express` / `plotly.graph_objects` | all interactive plots (`px.imshow` image grids, bars, heatmaps) |
| `scipy.ndimage.rotate`, `matplotlib` | rotation helper and plots in part 2 |
| `joblib` | saving and loading trained models (`classifier.joblib`, `mlp_fashionmnist_model.joblib`, `price_model.joblib`, `mlp_fashionmnist_rotated_model.joblib`, `model_rotation_regression.joblib`, `scaler.joblib`) |
| `otter-grader` | autograder checks (`grader.check("q…")`) |

---

## `fashion_pt_1.ipynb` — data wrangling, clustering, classifier, image augmentation

### Problem 0–1: Data as a DataFrame and class balance
- Each 28×28 image becomes a row in a pandas DataFrame with `image` (ndarray) and `label` (class name).
- Class balance is checked with `value_counts()` and `groupby('label').size()`: all 10 classes have exactly 6,000 images, so the set is balanced.
- `df.groupby('label').sample(2)` pulls two examples per class, shown with a `show_images` helper built on `px.imshow(facet_col=…)`.

### Problem 2: K-means on raw pixels
- Images are flattened to 784-vectors, a 1,000-image sample is taken, and `KMeans(n_clusters=10)` is fit.
- The `label × cluster` counts are shown as a stacked bar chart, and 7 random images per cluster are plotted.
- **Finding (written answer):** clusters follow silhouette and colour/brightness as well as clothing category. For example, one cluster mixes shoe types, and another mixes tops, because raw-pixel distance is dominated by overall shape and intensity.

### Problem 3: MLP classifier
- **Split:** `groupby('label').sample(frac=0.8)` gives a stratified 48,000 / 12,000 train/test split.
- **Preprocessing:** `StandardScaler` is fit on the training set only and applied to the test set. The notebook verifies the scaled training data has mean 0 and variance 1 per feature.
- **Model:** `MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=100, tol=1e-3)`.
- **Results:** train accuracy **0.991** vs test accuracy **0.885**, a clear overfitting gap. The confusion-matrix accuracy printed later is 0.881.
- **Per-class analysis:**
  - Trouser and Bag are the easiest classes.
  - **Shirt** is the hardest, with precision about 0.71 and recall about 0.70. It is confused with T-shirt, Pullover and Coat.
- **Hand-written confusion matrix:** built by looping over `test_df`. Precision and recall per class are derived from the diagonal and row/column sums.
- **Confidence analysis:** `predict_proba` gives a `confidence` column. The notebook then inspects:
  - the least-confident predictions,
  - correct but low-confidence "Ankle boot" images (a heel-like vs sneaker-like split in the class),
  - high-confidence wrong "Trouser → Dress" predictions.
  
  This is the main "failure case" visualisation in HW1.

### Problem 4: Image augmentation as matrix multiplication
Every augmentation is a **784×784 matrix** `T` applied as `image @ T.T`, so a transform is just one matmul and transforms can be composed.

| Transform | How the matrix is built |
|---|---|
| Vertical / horizontal flip | permutation matrix: `T[new_idx, old_idx] = 1` |
| Shift (dx, dy) | moves each in-bounds pixel; out-of-bounds pixels become zero |
| Blur (k×k) | each output pixel row holds `1/k²` weights over its neighbourhood |
| Rotation | forward mapping with nearest-neighbour rounding about the image centre. This leaves **holes** in the output |
| Bilinear rotation | *inverse* mapping: for each output pixel, rotate back to source coordinates and weight the 4 nearest source pixels, which removes the holes |
| `compose_transforms(*Ts)` | `T_total = T_n @ … @ T_1` starting from `np.eye(784)`, so a whole pipeline costs one matmul |

The notebook then applies shift, rotate, blur, rotate→blur and shift→rotate→blur to 100 test images and records the accuracy of the original MLP on each into `aug_performance`.
- **Best:** blur. It changes pixel values the least.
- **Worst:** shift→rotate→blur. It changes the flattened pixel vector the most.
- **Written answers:** order of transforms matters (matrix multiplication is associative but not commutative). A transform can be undone only when its matrix is invertible, so flips and rotations by exact multiples of 90° can be undone, while blur and cropping shifts lose information.

---

## `fashion_pt_2.ipynb` — regression, distribution shift, and three fixes

### Problem 5: Predicting price from pixels
- Joins a separate price CSV onto the train/test DataFrames and fits `LinearRegression` on raw pixels.
- Hand-rolled `compute_rmse` / `compute_mae` / `compute_r2` give about **RMSE 138, MAE 85, R² −0.40** (train and test are nearly identical).
- **Caveat I noticed:** the model is fit on the unscaled `X_train`, but the metrics cell calls `predict` on the *scaled* `X_train_sc` / `X_test_sc`. That input mismatch is a likely reason R² is negative.
- **Classification vs regression errors:** average relative price error is **1.47** for misclassified images vs **0.77** for correctly classified ones. This suggests images the classifier struggles with are also harder for the price model.

### Problem 6: The "secret" test set
- The original MLP scores 0.884 on the normal test split but only **0.205** on a hidden 2,000-image set.
- A class-wise accuracy bar chart and `confusion_matrix` show errors that look unlike the normal ones.
- Plotting misclassified images shows the secret images are all **rotated**. This is a distribution shift.

### Three solutions

| # | Approach | How it's implemented | Secret-set accuracy |
|---|---|---|---|
| Baseline | Original MLP on rotated data | — | **0.205** |
| 1 | **Rotation-invariant training** (augmentation) | Take 3,000 training images, rotate each 4× by `np.random.uniform(0, 360)` with `scipy.ndimage.rotate`, giving 12,000 images. Train a new `MLPClassifier` (100, 50) | **0.773** |
| 2 | **Predict then undo rotation** | `MLPRegressor((256, 128), early_stopping=True)` predicts the rotation angle from scaled pixels. Each image is then rotated by minus that angle and classified by the *original* MLP | **0.559** |
| 3 | **Test-time augmentation (TTA)** | Unrotate with the regressor, add 5 extra random rotations in ±70°, classify all variants, take the majority vote (`Counter`) | **0.4575** (printed) |

**Observations**
- Augmentation (solution 1) wins. It does not depend on a second model being accurate.
- Solution 2 is limited by the angle regressor. The printed sample predictions are off by roughly 6–28°, so residual rotation remains.
- In the angle-regression evaluation, the cell labels the output "Test MSE 0.89 / RMSE 0.94°", but the value comes from `model.score(...)`, which is **R²**, not MSE. The "0.94 degrees" figure is therefore not a real RMSE.
- The TTA result printed in the notebook (0.4575) is below the 58% target stated in the assignment text for this problem.
