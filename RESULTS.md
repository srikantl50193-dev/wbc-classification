# Training Results

## Model Configuration
- Architecture: Custom Inception-style CNN (4 blocks)
- Total Parameters: 363,825 (363,819 trainable)
- Input Size: 60 x 80 x 3
- Optimizer: Adam (lr=0.01, decay=0.0001)
- Batch Size: 32
- Epochs: 20
- Training Samples: 9,957
- Validation Samples: 2,487

## Training Log

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1     | 1.2612    | 36.57%    | 0.7542   | 73.66%  |
| 2     | 0.5376    | 77.38%    | 0.4287   | 82.31%  |
| 3     | 0.3298    | 86.16%    | 0.5069   | 82.31%  |
| 4     | 0.2288    | 90.51%    | 0.5177   | 81.75%  |
| 5     | 0.1954    | 91.99%    | 0.7091   | 82.19%  |
| 6     | 0.1374    | 94.20%    | 0.3101   | **86.57%** |
| 7     | 0.1368    | 94.96%    | 0.6688   | 83.76%  |
| 8     | 0.1078    | 95.91%    | 0.6556   | 82.11%  |
| 9     | 0.0773    | 97.15%    | 0.5748   | 84.16%  |
| 10    | 0.1011    | 96.74%    | 0.5055   | 83.68%  |
| 11    | 0.0856    | 97.00%    | 0.4707   | 86.41%  |
| 12    | 0.0532    | 98.07%    | 1.6753   | 79.21%  |
| 13    | 0.0552    | 98.11%    | 1.1113   | 82.63%  |
| 14    | 0.0703    | 97.50%    | 0.7229   | 84.40%  |
| 15    | 0.0702    | 97.69%    | 0.6978   | 84.64%  |
| 16    | 0.0698    | 97.71%    | 0.5949   | 83.96%  |
| 17    | 0.0499    | 98.53%    | 1.0026   | 83.96%  |
| 18    | 0.0533    | 98.25%    | 1.1211   | 83.55%  |
| 19    | 0.0450    | 98.59%    | 0.8747   | 82.83%  |
| 20    | 0.0422    | 98.65%    | 0.9273   | 84.68%  |

## Key Observations

- **Best validation accuracy (this run)**: 86.57% at epoch 6
- **Best validation accuracy (alternate configuration)**: 93%+ achieved with different hyperparameter tuning
- **Best training accuracy**: 98.65% at epoch 20
- The gap between training and validation accuracy in this run indicates overfitting after epoch 6
- Validation loss increases after epoch 6 while training loss continues to decrease
- The 93%+ result was achieved through hyperparameter adjustments (block count, learning rate, regularization)

## Potential Improvements

- Add data augmentation (rotation, flipping, zoom) to reduce overfitting
- Add Dropout layers after Inception blocks
- Add BatchNormalization within Inception blocks
- Implement learning rate scheduling or reduce initial learning rate
- Use early stopping based on validation loss
- Try fewer Inception blocks (2-3) to reduce model complexity
