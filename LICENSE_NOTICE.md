# License Notice

## This project's code

The original wrapper, runner, plotting, and report code in this
repository (everything outside `upstream/`) is released by the project
author for academic / educational use.

## Upstream RF-Diffusion

The contents of `upstream/RF-Diffusion/` are © their respective authors
(Chi et al., MobiCom 2024) and are released under the **GNU General
Public License v3.0**.

Key terms that apply to us:

- You may run the official code and weights for academic / research
  purposes.
- You must preserve the GPL-3.0 license notice.
- If you redistribute modified versions of the upstream code, the
  modifications must also be released under GPL-3.0.

This project does **not** redistribute modified upstream code; we use
the upstream code as a library and import its modules unmodified.

## Pretrained model and dataset

The pretrained weights under `upstream/RF-Diffusion/model/wifi/...` and
`upstream/RF-Diffusion/model/mimo/...`, plus the released dataset
samples under `upstream/RF-Diffusion/dataset/`, are made available by
the RF-Diffusion authors for research use under the same GPL-3.0
license.

## Paper citation

```
@inproceedings{chi2024rf,
  title={RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion},
  author={Chi, Guoxuan and Yang, Zheng and Wu, Chenshu and Xu, Jingao and
          Gao, Yuchong and Liu, Yunhao and Han, Tony Xiao},
  booktitle={Proceedings of the 30th Annual International Conference on
             Mobile Computing and Networking (MobiCom '24)},
  pages={77--92},
  year={2024}
}
```

The official artefact is at https://zenodo.org/records/10449052 and the
upstream code is at https://github.com/mobicom24/RF-Diffusion (commit
`eb872b0c4543da65424f5598ae40826e76e7edea`, 2024-01-08, captured in
`upstream/RF-Diffusion/.git/`).