# REvolve: Reward Evolution with Large Language Models using Human Feedback
******************************************************
**Official code release of our ICLR 2025 paper.**

<p align="center">
    <a href="https://rishihazra.github.io/REvolve/" target="_blank">
        <img alt="Documentation" src="https://img.shields.io/website/https/rishihazra.github.io/EgoTV?down_color=red&down_message=offline&up_message=link">
    </a>
    <a href="https://arxiv.org/abs/2406.01309" target="_blank">
        <img src="https://img.shields.io/badge/arXiv-2406.01309-red">
    </a>
    <a href="https://arxiv.org/pdf/2406.01309">
        <img src="https://img.shields.io/badge/Downloads-PDF-blue">
    </a>
</p>

<p align="center">
  <img src="revolve.gif" alt="egoTV">
</p>

## Setup
```shell
# clone the repository 
git clone https://github.com/RishiHazra/Revolve.git
cd Revolve
conda create -n "revolve" python=3.10
conda activate revolve
pip install -e .
```
## Notes

### 1) What `main.py` Does (Evolution → Training → Fitness) + Important Variables
`main.py` runs one full evolutionary generation. Critical variables inside it:
• `GEN_ID` - Current generation index. All individuals for this round are tagged GEN_ID_COUNTER.
• `baseline` - Experiment type (e.g., “revolve”). Controls database folder path.
• `run_id` - Numerical run identifier used as database/<baseline>/<run_id>/.
• `num_islands` - Number of islands (sub-populations).
• `population_size` - Number of individuals trained per generation.
• `num_episodes` - Number of evaluation episodes.
• `policy_training_steps` - Training length for each policy.

Workflow:
1. Generate reward functions for GEN_ID.
2. Save them under generated_fns/GEN_ID_COUNTER.txt.
3. Train all individuals in parallel (1 CPU core per policy).
4. During training (every 10k steps):
   - velocity_GEN_COUNTER.txt
   - SAC_GEN_COUNTER_STEP.zip
   - GEN_COUNTER_STEP.pkl
5. After training: evaluate → numeric fitness (environment-based, here is maximum velocity).
6. Save numeric fitness under fitness_scores/GEN_COUNTER.txt.


### 2) Running `visualize_best_individuals.py --generation <GEN>`
Purpose:
• Load velocity logs for GEN.
• Pick best checkpoint (highest forward velocity).
• Load model checkpoint + VecNormalize + reward function.
• Imporant to save as GEN_COUNTER.mp4 under videos_2x/. (recommended programme, simplescreenrecorder)

To run:
python3 visualize_best_individuals.py --generation <GEN> (run all individuals for that generation).


### 3) Running `ui_hf.py --gen_id <GEN>`
Purpose:
• Load all videos from human_feedback/videos_2x/.
• Randomize pairs and collect human choices.
• Save responses to human_feedback/generation_<GEN>/responses_<ID>.csv.

To run:
python3 ui_hf.py --gen_id <GEN>

### 4) Running `elo_scoring.py --gen_id <GEN>`
Purpose:
• Load ALL HF CSVs from generations 0..GEN.
• Compute ELO scores + normalize them.
• Save HF fitness to fitness_scores/GEN_COUNTER.txt.
• Save comments to human_feedback/GEN_COUNTER.txt.

To run:
python3 elo_scoring.py --gen_id <GEN>
replaces manual fitness scores (maximum velocity, to fitness score via elo scoring from human preferences)


### 5) Full Pipeline (GEN = current generation)
1. Train generation:
   python3 main.py
2. Visualize:
   python3 visualize_best_individuals.py --generation GEN
3. Collect HF:
   python3 ui_hf.py --gen_id GEN
4. Convert HF → fitness:
   python3 elo_scoring.py --gen_id GEN
5. Next generation:
   python3 main.py

### 6) Summary
• ui_hf.py groups HF by generation.
• elo_scoring.py accumulates HF 0..GEN and overwrites numeric fitness.
• visualize_best_individuals.py finds best checkpoints for GEN.
• Pipeline: TRAIN → VISUALIZE → RATE → SCORE → EVOLVE → repeat.


## Run
```shell
export ROOT_PATH='...'
export OPENAI_API_KEY='<your openai key>'
python main.py 

*Note, we will soon release the AirSim environment setup script.*

For AirSim, follow the instruction on this link [https://microsoft.github.io/AirSim/build_linux/](AirSim)
```shell
export AIRSIM_PATH='AirSim'
export AIRSIMNH_PATH='AirSimNH/AirSimNH/LinuxNoEditor/AirSimNH.sh'
```
 
## Other Utilities
* The prompts are listed in ```prompts``` folder.
* Elo scoring in ```human_feedback``` folder

## Citation

### To cite our paper:
```bibtex
@inproceedings{hazra2025revolve,
	title        = {{RE}volve: Reward Evolution with Large Language Models using Human Feedback},
	author       = {Rishi Hazra and Alkis Sygkounas and Andreas Persson and Amy Loutfi and Pedro Zuidberg Dos Martires},
	year         = 2025,
	booktitle    = {The Thirteenth International Conference on Learning Representations},
	url          = {https://openreview.net/forum?id=cJPUpL8mOw}
}
```
## License  
REvolve code is licensed under the [MIT license](https://github.com/RishiHazra/Revolve/blob/main/LICENSE).
