# ✅ Yoshi_Notebook_2.ipynb - READY FOR OVERNIGHT RUN

## 🎯 Final Status: COMPLETE & READY

All changes have been implemented successfully. The notebook is now optimized for an overnight CPU run.

## 📊 Notebook Structure (38 cells total)

### Setup & Preparation (Cells 0-29)
- **Cell 0**: Runtime estimate header (12-20 hours total)
- **Cells 1-14**: Library imports, device setup, model loading (ViT-GPT2)
- **Cells 15-22**: Attention extraction and visualization functions
- **Cells 23-29**: Attack pipeline setup, candidate pixel selection, DE algorithm

### Differential Evolution Attack (Cells 30-33)
- **Cell 30**: Optimized fitness function (ε=15, no exploration bonus)
- **Cell 31**: 🆕 **DEBUG TESTS** - Validates fitness function before running
- **Cell 32**: DE attack execution (pop=150, gen=80, ~10-16 hours)
- **Cell 33**: DE results evaluation (BLEU scores, transferability)

### PGD Attack (Cells 34-36)
- **Cell 34**: PGD section header
- **Cell 35**: PGD attack implementation (iter=100, α=2.0, ~2-4 hours)
- **Cell 36**: PGD results evaluation (BLEU scores, transferability)

### Comparison (Cell 37)
- **Cell 37**: Side-by-side comparison with 2×3 visualization grid

## 🔧 Key Parameters

### Differential Evolution
- **Candidate pixels**: 1,000 (reduced from 30,000)
- **Search space**: 3,000 dimensions (manageable)
- **Population size**: 150
- **Generations**: 80
- **Epsilon**: 15
- **Expected runtime**: 10-16 hours
- **Checkpoints**: Every 20 generations → `./checkpoints/de_best_gen{N}.npy`

### PGD
- **Candidate pixels**: 1,000 (same as DE)
- **Iterations**: 100
- **Alpha (step size)**: 2.0
- **Epsilon**: 15
- **Expected runtime**: 2-4 hours
- **Checkpoints**: Every 20 iterations → `./checkpoints/pgd_iter{N}.npy`

## 🚀 What to Expect When Running

### Cell 31 Output (Debug Tests)
You'll see 3 tests:
1. **Zero perturbation**: Should return fitness ≈ 0
2. **Random perturbation**: Should show if random noise can change caption
3. **Manual comparison**: Demonstrates fitness calculation logic

This helps verify the fitness function works correctly before the long DE run.

### Cell 32 Output (DE Attack)
Every 5 generations, you'll see:
```
--- Generation 5/80 ---
Best fitness: 12.3456
Best caption so far: 'a dog sitting on a bench'
```

Every 20 generations: `Checkpoint saved: ./checkpoints/de_best_gen20.npy`

### Cell 35 Output (PGD Attack)
Every 10 iterations:
```
--- Iteration 10/100 ---
Loss: 3.2145
Current caption: 'a car on the street'
```

Every 20 iterations: `Checkpoint saved: ./checkpoints/pgd_iter20.npy`

### Cell 37 Output (Final Comparison)
A formatted table:
```
Method     | ViT BLEU↓   | BLIP BLEU↓  | Transfer  | Avg L∞  | Max L∞
----------------------------------------------------------------------
DE         |   85.3%     |   42.1%     |  49.4%   |  12.45  |  15.00
PGD        |   91.2%     |   38.7%     |  42.4%   |  11.23  |  15.00
```

Plus a 2×3 visualization grid showing original, adversarial, and perturbation maps.

## 📁 Output Files

All checkpoints will be saved to `./checkpoints/`:
- `de_best_gen20.npy`, `de_best_gen40.npy`, `de_best_gen60.npy`, `de_best_gen80.npy`
- `pgd_iter20.npy`, `pgd_iter40.npy`, `pgd_iter60.npy`, `pgd_iter80.npy`, `pgd_iter100.npy`

## ✅ What Was Changed

1. ✅ Reduced candidate pixels: 30K → 1K
2. ✅ Fixed fitness function (removed exploration bonus)
3. ✅ Added fitness debugging cell
4. ✅ Optimized DE parameters (pop=150, gen=80)
5. ✅ Added checkpointing to DE
6. ✅ Implemented simplified PGD with proper loss maximization
7. ✅ Added PGD evaluation
8. ✅ Added side-by-side comparison
9. ✅ Removed duplicate function definitions
10. ✅ Removed commented code blocks

## 🎬 How to Run

1. Open the notebook in your environment (Jupyter/Colab/VSCode)
2. Run all cells sequentially (or "Run All")
3. Leave it running overnight
4. Check back in 12-20 hours for complete results

## ⚠️ Important Notes

- **CPU Runtime**: The notebook is configured for CPU. If you have GPU available, it will use it automatically and run faster.
- **Checkpoints**: Don't delete the `./checkpoints/` directory while running - it's your backup!
- **Memory**: With 1,000 pixels, memory usage should be reasonable (~2-4 GB)
- **Interruption**: If interrupted, you can resume from the last checkpoint (though the notebook doesn't auto-resume - you'd need to load manually)

## 📈 Expected Results

Based on the optimized parameters:
- **DE**: Should achieve 30-80% BLEU decrease on ViT-GPT2
- **PGD**: Should achieve 40-90% BLEU decrease (potentially more effective due to gradients)
- **Transferability**: 20-50% transfer to BLIP expected
- **Perturbations**: Average L∞ around 10-14 per pixel (within ε=15 bound)

## 🎉 You're All Set!

The notebook is **ready to run**. Good luck with your overnight experiment! 🚀

---
*Last updated: 2025-11-17*
*Configuration: 1000 pixels, ε=15, DE(150,80), PGD(100,2.0)*
