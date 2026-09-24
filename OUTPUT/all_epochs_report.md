# All epoch results — FINAL

## Stage 3 baseline (EfficientNet-B4, 20 epochs)
- epoch 1/20: train_loss=0.2199, val_acc=0.9656
- epoch 2/20: train_loss=0.0524, val_acc=0.9531
- epoch 3/20: train_loss=0.0235, val_acc=0.9906
- epoch 4/20: train_loss=0.0129, val_acc=0.9719
- epoch 5/20: train_loss=0.0122, val_acc=0.9688
- epoch 6/20: train_loss=0.0213, val_acc=0.9625
- epoch 7/20: train_loss=0.0041, val_acc=0.9938
- epoch 8/20: train_loss=0.0015, val_acc=0.9875
- epoch 9/20: train_loss=0.0008, val_acc=0.9656
- epoch 10/20: train_loss=0.0076, val_acc=0.9938
- epoch 11/20: train_loss=0.0055, val_acc=0.9906
- epoch 12/20: train_loss=0.0018, val_acc=0.9906
- epoch 13/20: train_loss=0.0007, val_acc=0.9719
- epoch 14/20: train_loss=0.0003, val_acc=0.9812
- epoch 15/20: train_loss=0.0001, val_acc=0.9688
- epoch 16/20: train_loss=0.0006, val_acc=0.9906
- epoch 17/20: train_loss=0.0003, val_acc=0.9969
- epoch 18/20: train_loss=0.0002, val_acc=0.9875
- epoch 19/20: train_loss=0.0002, val_acc=0.9906
- epoch 20/20: train_loss=0.0001, val_acc=1.0000

## Stage 4 full sweep (135 epochs, 10 frontier configs / 9 trained runs — shared control trained once)
### skin_lam=0.0_illum_lam=0.0 (15 epochs)
e1=2.06 e2=2.54 e3=2.40 e4=2.51 e5=2.61 e6=2.25 e7=1.97 e8=2.54 e9=2.31 e10=2.54 e11=2.56 e12=2.38 e13=3.62 e14=2.60 e15=2.29
### skin_lam=0.25_illum_lam=0.0 (15 epochs)
e1=3.40 e2=3.36 e3=2.72 e4=2.93 e5=2.55 e6=3.64 e7=3.67 e8=2.79 e9=2.97 e10=3.40 e11=2.83 e12=2.77 e13=3.12 e14=3.29 e15=3.30
### skin_lam=0.5_illum_lam=0.0 (15 epochs)
e1=3.10 e2=3.27 e3=2.65 e4=3.04 e5=3.27 e6=3.88 e7=3.39 e8=3.44 e9=3.02 e10=2.85 e11=3.02 e12=2.67 e13=2.79 e14=2.95 e15=3.11
### skin_lam=1.0_illum_lam=0.0 (15 epochs)
e1=2.78 e2=2.61 e3=3.17 e4=2.69 e5=3.04 e6=2.83 e7=2.65 e8=3.17 e9=3.00 e10=3.02 e11=3.08 e12=2.32 e13=2.61 e14=2.71 e15=2.59
### skin_lam=2.0_illum_lam=0.0 (15 epochs)
e1=3.01 e2=3.08 e3=3.22 e4=3.18 e5=3.13 e6=3.08 e7=2.98 e8=2.72 e9=2.76 e10=2.79 e11=3.26 e12=2.58 e13=2.35 e14=2.68 e15=3.05
### skin_lam=0.0_illum_lam=0.25 (15 epochs)
e1=3.34 e2=2.13 e3=2.60 e4=2.47 e5=2.26 e6=2.81 e7=2.23 e8=2.54 e9=2.94 e10=2.32 e11=3.14 e12=2.59 e13=2.51 e14=2.26 e15=2.48
### skin_lam=0.0_illum_lam=0.5 (15 epochs)
e1=2.79 e2=2.70 e3=2.51 e4=3.95 e5=3.19 e6=2.82 e7=2.24 e8=3.19 e9=4.43 e10=3.22 e11=2.98 e12=2.98 e13=2.75 e14=2.31 e15=3.63
### skin_lam=0.0_illum_lam=1.0 (15 epochs)
e1=3.18 e2=2.99 e3=3.75 e4=2.94 e5=2.49 e6=2.62 e7=2.51 e8=2.76 e9=2.83 e10=3.14 e11=2.24 e12=2.39 e13=2.64 e14=2.76 e15=2.76
### skin_lam=0.0_illum_lam=2.0 (15 epochs)
e1=2.22 e2=2.18 e3=1.99 e4=2.65 e5=2.64 e6=2.67 e7=2.65 e8=2.43 e9=2.44 e10=2.89 e11=2.34 e12=2.76 e13=2.54 e14=2.49 e15=2.41

## Stage 4 finished-run validation metrics (9/9)
- skin_lam=0.0_illum_lam=0.0: fake_acc=0.9906, skin_probe_acc=0.5594, illum_probe_acc=0.1938
- skin_lam=0.25_illum_lam=0.0: fake_acc=0.95, skin_probe_acc=0.5625, illum_probe_acc=0.1656
- skin_lam=0.5_illum_lam=0.0: fake_acc=0.9688, skin_probe_acc=0.5656, illum_probe_acc=0.1562
- skin_lam=1.0_illum_lam=0.0: fake_acc=0.9781, skin_probe_acc=0.5656, illum_probe_acc=0.1875
- skin_lam=2.0_illum_lam=0.0: fake_acc=0.975, skin_probe_acc=0.5625, illum_probe_acc=0.1625
- skin_lam=0.0_illum_lam=0.25: fake_acc=0.9688, skin_probe_acc=0.5625, illum_probe_acc=0.1656
- skin_lam=0.0_illum_lam=0.5: fake_acc=0.9781, skin_probe_acc=0.5625, illum_probe_acc=0.2156
- skin_lam=0.0_illum_lam=1.0: fake_acc=1.0, skin_probe_acc=0.5625, illum_probe_acc=0.1656
- skin_lam=0.0_illum_lam=2.0: fake_acc=0.9594, skin_probe_acc=0.5625, illum_probe_acc=0.1594