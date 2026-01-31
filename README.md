# Coarse-to-Real: Generative Rendering for Populated Dynamic Scenes

> **Status:** 🚧 Work in Progress — code and models will be released shortly.

## Abstract
Traditional rendering pipelines rely on complex assets, accurate materials and lighting, and substantial computational resources to produce realistic imagery, yet they still face challenges in scalability and realism for populated dynamic scenes. 
We present C2R (Coarse-to-Real), a generative rendering framework that synthesizes real-style urban crowd videos from coarse 3D simulations. Our approach uses coarse 3D renderings to explicitly control scene layout, camera motion, and human trajectories, while a learned neural renderer generates realistic appearance, lighting, and fine-scale dynamics guided by text prompts. 
To overcome the lack of paired training data between coarse simulations and real videos, we adopt a two-phase mixed CG–real training strategy that learns a strong generative prior from large-scale real footage and introduces controllability through shared implicit spatio-temporal features across domains. 
The resulting system supports coarse-to-fine control, generalizes across diverse CG and game inputs, and produces temporally consistent, controllable, and realistic urban scene videos from minimal 3D input.

The project aims to provide a **reproducible, research-oriented implementation** together with pretrained models and clear usage instructions.
At the moment, the repository serves as a **placeholder and roadmap**. The full implementation, training code, and inference scripts are currently under preparation and will be published soon.

---

## Planned Features & Roadmap

The following items outline what will be included in the first public release:

* [ ] Core Python implementation
* [ ] Model architecture definition
* [ ] Training pipeline
* [ ] Inference / evaluation scripts
* [ ] Pretrained model weights
* [ ] Example inputs and outputs
* [ ] Reproducibility instructions (seeds, configs)
* [ ] Documentation and usage examples

---

## Installation (Coming Soon)

Once released, the project will be installable using standard Python tooling.

```bash
# Clone the repository
git clone https://github.com/GonzaloGNogales/coarse2real.git
cd coarse2real

# Create a conda environment (recommended)
conda create -n coarse2real python=3.10 -y
conda activate coarse2real


# Install dependencies
pip install -r requirements.txt
```

> ⚠️ The `requirements.txt` file and dependencies will be finalized and published together with the code.

---

## Usage (Coming Soon)

Example commands that will be supported:

```bash
# Training
python train.py --config configs/train.yaml

# Inference
python inference.py --input data/example --output results/
```

Detailed explanations of configuration options, expected inputs, and outputs will be provided in the final release.

---

## Project Structure (Planned)

```text
.
├── configs/        # Training and inference configurations
├── data/           # Example or processed data
├── models/         # Model definitions
├── scripts/        # Training / inference scripts
├── utils/          # Helper functions
├── requirements.txt
└── README.md
```

---

## License

This project is licensed under Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International (CC BY-NC-ND 4.0).

### Allowed
- Sharing the original work with attribution
- Using it for research/education (non-commercial)

### Not allowed
- Commercial use
- Redistribution of modified versions

See the [LICENSE](./LICENSE) file for details.

---

## Citation

If you use this work in academic research, a citation entry will be provided once the project is officially released.

---

## Contact

For questions, collaborations, or commercial licensing inquiries, please contact:

* **Name:** Yi Zhou
* **Email:** [[yizhou@roblox.com](mailto:yizhou@roblox.com)]
* **Email:** [[zhouyisjtu2012@gmail.com](mailto:zhouyisjtu2012@gmail.com)]


* **Name:** Gonzalo Gomez-Nogales
* **Email:** [[gonzalo.gomez@urjc.es](mailto:gonzalo.gomez@urjc.es)]

---

✨ *Thank you for your interest. Stay tuned — the full release is coming soon!*
