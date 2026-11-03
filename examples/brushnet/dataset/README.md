# SFU ROI QP0 + BG train/test PNGs

Source: output/postprocess_sfu_full_qp0_bg_multiqp/03_png.

Each bgQP directory is a separate dataset. Train contains the first 80% of
frames from each sequence, and test contains the last 20%. The directory
layout matches SFU_STC_flow:

    bgQP/
      train/
        input/Class_*/Sequence/000000.png
        GT/Class_*/Sequence/000000.png
        mask/Class_*/Sequence/000000.png
      test/
        input/Class_*/Sequence/000000.png
        GT/Class_*/Sequence/000000.png
        mask/Class_*/Sequence/000000.png
The QP-specific train.csv and test.csv list each input image and its original
frame number.

Original GT frames were converted directly from the 8-bit yuv420p source YUV
in Data/SFU. Masks were rendered from the full ROI descriptors: 255 (white)
means ROI and 0 (black) means background. Matching GT and mask images across
QPs are hardlinks to the same underlying files; the temporary _shared
directory was removed.

Use train_pairs.csv and test_pairs.csv for supervised training. Each row gives
relative input, gt and mask paths for exactly the same source frame. All
images in the QP directories are hardlinks, so the QP copies use no second
copy of image data on disk.

| BG QP | Sequences | Train | Test |
| --- | ---: | ---: | ---: |
| 46 | 14 | 4,824 | 1,206 |
| 52 | 11 | 3,544 | 886 |
| 56 | 14 | 4,824 | 1,206 |
| 60 | 14 | 4,824 | 1,206 |

QP52 excludes BasketballDrill, BQSquare, and BlowingBubbles because this
dataset uses only full-sequence PNGs from 03_png. Their partial QP52 clips
remain in output/postprocess_sfu_full_qp0_bg_multiqp/04_partial_qp52.

Rebuild with: python Plot/create_sfu_qp0_bg_train_test.py

Rebuild GT and masks with: python Plot/add_sfu_gt_mask_to_split.py

Restructure an older split with: python Plot/restructure_sfu_qp0_bg_split.py
