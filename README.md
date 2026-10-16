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

MuJoCo simulation, full terrain map, and live Viser viewer. No Isaac installation
is required. Use Python 3.11. **Checkpoint release is pending.**

```bash
git clone https://github.com/osheraz/vigor.git
cd vigor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To run the demo, supply a compatible TorchScript policy as `checkpoints/student.pt`
with its matching `checkpoints/policy.json`.

```bash
python sim.py --start-mode random   # Random motion phase and terrain placement (default)
python sim.py --start-mode settled  # Random lying pose, physically settled before starting
```

Files: `sim.py` (simulation/viewer), `starts.py` (start sampling), `assets/`
(robot, terrain, motions), and `checkpoints/` (policy configuration).

### TODO

- [ ] Pretrained checkpoint release
- [ ] Teacher training and motion preparation
- [ ] Student training
- [ ] Full integration with [g1_deploy (refactor branch)](https://github.com/osheraz/g1_deploy/tree/proposal/reorganization-20261031) for real-world deployment

We are developing a minimal onboard version.

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
