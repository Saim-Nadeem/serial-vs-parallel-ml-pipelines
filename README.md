# Serial vs Parallel ML Pipelines for Binary Classification

This project benchmarks the performance of **serial** and **parallel** machine learning pipelines using structured tabular data. It compares model training speed and accuracy across multiple compute strategies: **single-core CPU**, **multi-core CPU**, **distributed clusters (Dask)**, and **GPU acceleration (TensorFlow)**.

Two model families are analyzed:
- **XGBoost**: Serial vs. Distributed (via Dask)
- **Neural Networks**: Single-core vs. Multi-core vs. GPU

---

## Dataset

- **File**: [`pdc_dataset_with_target.csv`](pdc_dataset_with_target.csv)  
- **Type**: Tabular data with numerical and categorical features  
- **Target**: Binary classification

---

## Objectives

- Evaluate the speedup of parallel pipelines over serial ones  
- Maintain high classification accuracy  
- Analyze performance tradeoffs between CPU, GPU, and distributed processing

---

## Preprocessing Pipeline

The same preprocessing steps are applied in all experiments:

- Remove duplicates  
- Impute missing values  
  - Categorical: Mode  
  - Numerical: Mean  
- Remove outliers  
- One-Hot Encode categorical variables  
- Standardize features with `StandardScaler`

---

## Model Implementations

### XGBoost

- **Serial Training**  
  [`PDC_XGBoost_Sequential.ipynb`](PDC_XGBoost_Sequential.ipynb)  
  → Basic CPU-based training using `XGBClassifier`

- **Parallel (Distributed) Training**  
  [`PDC_XGBoost_Distributed.ipynb`](PDC_XGBoost_Distributed.ipynb)  
  → Uses Dask + Coiled to distribute training across multiple workers

---

### Neural Networks (TensorFlow)

- **Single-core CPU**  
  [`pdc-neuralnetwork-singlecore.ipynb`](pdc-neuralnetwork-singlecore.ipynb)

- **Multi-core CPU**  
  [`pdc-neuralnetwork-multicore.ipynb`](pdc-neuralnetwork-multicore.ipynb)

- **GPU**  
  [`pdc-neuralnetwork-gpu.ipynb`](pdc-neuralnetwork-gpu.ipynb)

---

## Evaluation Metrics

Each model is evaluated using:

- Accuracy  
- F1 Score  
- Confusion Matrix  
- Training Time (seconds)

---

## Results Summary

| Model Variant                  | Speedup Compared to Serial |
|-------------------------------|----------------------------|
| XGBoost (CPU → Dask)          | 91%                        |
| Neural Network (1-Core → GPU) | 70%                        |
| Neural Network (1 → Multi)    | 39%                        |
| Neural Network (Multi → GPU)  | 45%                        |

All model variants achieved over 60% accuracy.

---

## Analysis

### XGBoost
- Suitable for structured/tabular data  
- Dask scales well for larger datasets  
- Less expressive for complex patterns

### Neural Networks
- Highly expressive and flexible  
- GPU acceleration offers major speedups  
- More expensive to train on CPU  
- Requires careful hyperparameter tuning

---

## Experimental Setup

### Hardware
- CPU: Intel multi-core (local)  
- GPU: NVIDIA P100 (Kaggle)

### Software
- Python 3.13  
- Libraries:  
  - `scikit-learn`, `pandas`, `numpy`  
  - `xgboost`, `dask`, `coiled`  
  - `tensorflow`

### Environments
- Local Jupyter Notebook  
- Kaggle Notebooks  

---

## License

This project is licensed under the [MIT License](LICENSE).  
You may use, modify, and distribute it with attribution.

---

## Author

[Saim-Nadeem](https://github.com/Saim-Nadeem)

---

## Contributing

Contributions are welcome:

1. Fork this repository  
2. Create a new branch (`git checkout -b feature-name`)  
3. Commit your changes (`git commit -m "Add feature"`)  
4. Push to your fork (`git push origin feature-name`)  
5. Open a Pull Request
