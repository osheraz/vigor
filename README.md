## VIGOR

[[project page](https://vigor2026.github.io)] [[arxiv](https://arxiv.org/abs/2602.16511)]

<table><tr><td>
  <strong>Unified Perceptive Fall Safety for Humanoids.</strong><br />
  <small>
  Osher&nbsp;Azulay<sup>1</sup>, Zhengjie&nbsp;Xu<sup>1</sup>, Andrew&nbsp;Scheffer<sup>1</sup>, Stella&nbsp;X.&nbsp;Yu<sup>1</sup>
  </small>
</td></tr>
</table>

<sup>1</sup><em>University of Michigan</em>

### MuJoCo demo

MuJoCo simulation, full terrain map, and live Viser viewer. Install
[uv](https://docs.astral.sh/uv/) and [Git LFS](https://git-lfs.com).
No Isaac installation is required.

```bash
git lfs install
git clone https://github.com/osheraz/vigor.git
cd vigor
git lfs pull
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

```bash
python sim.py --model 29 --start-mode settled
python sim.py --model 23 --start-mode settled
```

Use `--start-mode random` to sample a random motion phase and terrain placement.
Model 29 is the default. Episodes last 7.5 seconds.

### TODO

- [ ] Teacher training and motion preparation
- [ ] Student training
- [ ] Full integration with [g1_deploy](https://github.com/osheraz/g1_deploy) for real-world deployment

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
