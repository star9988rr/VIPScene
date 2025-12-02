<h2 align="center">
    VIPScene<br>
    Video Perception Models for 3D Scene Synthesis<br>
</h2>

<h4 align="center">
  <a href="https://openreview.net/pdf?id=D0YNbanYfB"><i>Paper</i></a> | <a href="https://vipscene.github.io/"><i>Project Page</i></a>
</h4>

VIPScene is a novel framework that exploits the encoded commonsense knowledge of 3D physical world in video generation models to ensure coherent scene layouts and consistent object placements across views. VIPScene accepts both text and image prompts and seamlessly integrates video generation, feedforward 3D reconstruction, and open-vocabulary perception models to semantically and geometrically analyze each object in a scene.

## Core Dependencies & External Tools
VIPScene is built upon the [Holodeck](https://github.com/allenai/Holodeck) codebase and [AI2-THOR](https://ai2thor.allenai.org/ithor/documentation/#requirements) (which supports visualization in Unity). It seamlessly integrates the following state-of-the-art tools for its perception framework:

- **[UniDepth](https://github.com/lpiccinelli-eth/UniDepth)**: Universal Monocular Metric Depth Estimation
- **[Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)**: Grounded Segment Anything for open-vocabulary object segmentation
- **[Fast3R](https://github.com/facebookresearch/fast3r)**: Fast 3D Reconstruction
- **[Mast3r](https://github.com/naver/mast3r)**: 3D Reconstruction from Multiple Views for scene understanding

## Installation

1. **Clone and Install VIPScene**
   ```bash
   git clone https://github.com/VIPScene/VIPScene.git
   cd VIPScene
   conda create --name vipscene python=3.10
   conda activate vipscene
   pip install -r requirements.txt
   pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+8524eadda94df0ab2dbb2ef5a577e4d37c712897
   ```

2. **Setup External Tools**
   Please refer to the respective repositories linked above ([UniDepth](https://github.com/lpiccinelli-eth/UniDepth), [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [Fast3R](https://github.com/facebookresearch/fast3r), [Mast3r](https://github.com/naver/mast3r)) for their specific installation instructions and environment setup.

## Data
Please refer to [Holodeck](https://github.com/allenai/Holodeck) for data download instructions.

## Usage

**Prerequisite:** Our system uses `gpt-4o-2024-05-13`. Please ensure you have an OpenAI API key with access to this model. You can set it as an environment variable:
```bash
export OPENAI_API_KEY=<your_key>
```

For detailed step-by-step instructions on running the full video processing pipeline (including object analysis, depth estimation, segmentation, and 3D reconstruction), please refer to the **[Usage Documentation](docs/usage_pipeline.md)**.


## Citation
Please cite the following paper if you use this code in your work.

```bibtex
@article{huang2025video,
  title={Video Perception Models for 3D Scene Synthesis},
  author={Huang, Rui and Zhai, Guangyao and Bauer, Zuria and Pollefeys, Marc and Tombari, Federico and Guibas, Leonidas and Huang, Gao and Engelmann, Francis},
  journal={arXiv preprint arXiv:2506.20601},
  year={2025}
}
```

## Acknowledgement
We would like to express our sincere gratitude to the authors of [Holodeck](https://github.com/allenai/Holodeck), [UniDepth](https://github.com/lpiccinelli-eth/UniDepth), [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [Fast3R](https://github.com/facebookresearch/fast3r), and [Mast3R](https://github.com/naver/mast3r) for their excellent open-source work.
