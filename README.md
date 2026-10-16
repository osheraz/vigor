## VIGOR

[[project page](https://vigor2026.github.io)] [[arxiv](https://arxiv.org/abs/2602.16511)]

Code release for our paper:

<table><tr><td>
  <strong>Unified Perceptive Fall Safety for Humanoids.</strong><br />
  <small>
  Osher&nbsp;Azulay<sup>1</sup>, Zhengjie&nbsp;Xu<sup>1</sup>, Andrew&nbsp;Scheffer<sup>1</sup>, Stella&nbsp;X.&nbsp;Yu<sup>1</sup>
  </small>
</td></tr>
</table>

<sup>1</sup><em>University of Michigan</em>

### MuJoCo demo

Pretrained policy, full terrain map, and live Viser viewer. No Isaac installation
is required. Use Python 3.11 and install [Git LFS](https://git-lfs.com/).

```bash
git lfs install
git clone https://github.com/osheraz/vigor.git
cd vigor
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python sim.py
```

```bash
python sim.py --start-mode random   # Random motion phase and terrain placement (default)
python sim.py --start-mode settled  # Random lying pose, physically settled before starting
```

Files: `sim.py` (simulation/viewer), `starts.py` (start sampling), `assets/`
(robot, terrain, motions), and `checkpoints/` (policy and configuration).

### TODO

- [ ] Teacher training and motion preparation
- [ ] Student training
- [ ] Real-world deployment code

## BibTeX

```bibtex
@misc{azulay2026vigor,
  title        = {Unified Perceptive Fall Safety for Humanoids},
  author       = {Osher Azulay and Zhengjie Xu and Andrew Scheffer and Stella X. Yu},
  year         = {2026},
  eprint       = {2602.16511},
  archivePrefix= {arXiv},
  primaryClass = {cs.RO},
  url          = {https://arxiv.org/abs/2602.16511}
}
```
