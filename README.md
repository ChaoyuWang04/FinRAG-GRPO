# AdCampaignAgent-SFT

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ChaoyuWang04/FinRAG-GRPO">
    <img src="images/logo.jpg" alt="Logo" width="100" height="100">
  </a>

<h3 align="center">AdCampaignAgent-SFT</h3>

  <p align="center">
    A rule-based synthetic SFT dataset for training tool-calling agents in mobile game user acquisition (UA) — grounded in real ad operations, ROAS/Retention safety baselines, and multi-turn reasoning chains.
    <br />
    <a href="https://github.com/ChaoyuWang04/AdCampaignAgent-SFT"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://huggingface.co/datasets/SamWang0405/AdCampaignAgent-SFT">🤗 HuggingFace Dataset</a>
    &middot;
    <a href="https://github.com/ChaoyuWang04/AdCampaignAgent-SFT/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/ChaoyuWang04/AdCampaignAgent-SFT/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

AdCampaignAgent-SFT is an open-source pipeline for generating high-quality **Supervised Fine-Tuning (SFT)** datasets targeting LLM-based tool-calling agents in the mobile game advertising domain. It produces structured multi-turn dialogues that reflect real UA (User Acquisition) workflows — from campaign performance analysis to creative asset uploads — grounded in business-realistic ROAS and Retention safety baselines.

**Dataset highlights (`AdCampaignAgent-SFT`):**

| Metric | Value |
|--------|-------|
| Total conversations | 1,000 |
| Format | OpenAI Messages (`role` / `content` / `tool_calls` / `tool_call_id`) |
| Unique tools | 15 |
| Distinct scene tags | 22 |
| Platforms covered | Google UAC · Meta · TikTok · AppLovin · Unity |
| Game genres | Casual · Puzzle · Hyper-casual · RPG · Strategy |
| Avg turns / conversation | 7.2 messages |
| With tool calls | 900 (90.0%) |
| With clarification turns | 96 (9.6%) |
| Refusal conversations | 100 (10.0%) |
| Format errors | ✅ 0 — clean |

The generation pipeline is fully **rule-based** — no LLM API calls required. Each conversation is constructed from a seed record (workflow type, scene tag, ROAS/Retention baselines) through deterministic mock tool execution, ensuring internal data consistency across all tool results within a single dialogue.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

[![Python][Python-badge]][Python-url]
[![HuggingFace][HuggingFace-badge]][HuggingFace-url]
[![Pandas][Pandas-badge]][Pandas-url]
[![Matplotlib][Matplotlib-badge]][Matplotlib-url]
[![Seaborn][Seaborn-badge]][Seaborn-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

- Python 3.10+
- pip

```sh
pip install matplotlib seaborn pandas tqdm
```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ChaoyuWang04/AdCampaignAgent-SFT.git
   cd AdCampaignAgent-SFT
   ```

2. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```

3. Verify project structure
   ```
   AdCampaignAgent-SFT/
   ├── checker/               # auto-generated quality reports and figures
   │   ├── ad_agent_sft_*_cn_message/
   │   └── ad_agent_sft_*_cn_sharegpt/
   ├── data/                  # generated seeds and SFT datasets
   │   ├── ad_agent_seeds_*_cn.json
   │   ├── ad_agent_sft_*_cn_message.json
   │   └── ad_agent_sft_*_cn_sharegpt.json
   ├── docs/
   │   └── 0_Summary.md
   ├── images/
   │   ├── logo.png
   │   └── screenshot.png
   ├── src/
   │   ├── 1.1_ad_gen_data_cn.py
   │   ├── 1.2_ad_gen_data_en.py
   │   ├── 2.1_convert_dataset_sharegpt_cn.py
   │   ├── 2.2_convert_dataset_sharegpt_en.py
   │   ├── 2.3_convert_data_message_cn.py
   │   ├── 2.4_convert_data_message_en.py
   │   └── 3_dataquality_check.py
   ├── tools/
   │   └── 0_all_tools.json
   ├── LICENSE
   └── README.md
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

The pipeline runs in three sequential stages:

**Stage 1 — Generate seed records**

Produces 1,000 structured seed records covering 7 workflows and 22 scene tags, with pre-computed ROAS/Retention baselines per platform × genre.

```sh
python src/1_ad_gen_data.py
# Output: data/ad_agent_seeds_<timestamp>.json
```

**Stage 2 — Generate conversations**

Converts each seed into a full multi-turn dialogue in OpenAI Messages format. Tool results are mock-generated deterministically from the seed's `scene_tag` and baselines, ensuring all metrics within a single conversation are internally consistent.

```sh
python src/2_ad_gen_conversation.py data/ad_agent_seeds_<timestamp>.json data/ad_agent_sft_<timestamp>.json
# Output: data/ad_agent_sft_<timestamp>.json
```

**Stage 3 — Quality analysis & report**

Auto-detects format (OpenAI Messages or ShareGPT), runs full quality checks, generates 6 figures and a `dataset_card.md` ready for HuggingFace upload.

```sh
python src/3_analyze_dataset.py
# Input JSON file name: ad_agent_sft_<timestamp>.json
# Output: checker/ad_agent_sft_<timestamp>/
#           ├── dataset_card.md
#           ├── fig1_workflow.png
#           ├── fig2_scene.png
#           ├── fig3_turn_dist.png
#           ├── fig4_token_dist.png
#           ├── fig5_tool_freq.png
#           └── fig6_platform_genre.png
```

**Quick format validation (no full report)**

```sh
python -c "
import json
data = json.load(open('data/ad_agent_sft_<timestamp>.json'))
sample = data[50]['messages']
for m in sample:
    print(f\"{m['role']:12} | tool_calls={'tool_calls' in m} | tool_call_id={'tool_call_id' in m}\")
"
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] 7-workflow seed generation with rule-based scene tagging
- [x] 15-tool mock executor with internally consistent ROAS/Retention metrics
- [x] OpenAI Messages format with strict `tool_call_id` pairing
- [x] Auto-format detection (OpenAI Messages / ShareGPT)
- [x] Automated quality report with 6 figures + Markdown dataset card
- [ ] Increase `validate_fail` scenes from 17 → 50+ samples
- [ ] Add `tool_call` last-turn samples (~20 conversations)
- [ ] Balance `industry_benchmark` and `platform_policy` knowledge scenes to 40+ each
- [ ] Extend to English-only and bilingual variants
- [ ] Fine-tuned model checkpoint (Qwen3-0.6B on AdCampaignAgent-SFT)

See the [open issues](https://github.com/ChaoyuWang04/AdCampaignAgent-SFT/issues) for a full list of proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/ChaoyuWang04/AdCampaignAgent-SFT/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ChaoyuWang04/AdCampaignAgent-SFT" alt="contrib.rocks image" />
</a>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Chaoyu Wang - [![LinkedIn][linkedin-shield]][linkedin-url] - [GitHub](https://github.com/ChaoyuWang04)

Project Link: [https://github.com/ChaoyuWang04/AdCampaignAgent-SFT](https://github.com/ChaoyuWang04/AdCampaignAgent-SFT)

Dataset Link: [https://huggingface.co/datasets/SamWang0405/AdCampaignAgent-SFT](https://huggingface.co/datasets/SamWang0405/AdCampaignAgent-SFT)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) — SFT training framework reference
* [Qwen3](https://huggingface.co/Qwen) — base model for fine-tuning experiments
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template) — README structure

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/ChaoyuWang04/AdCampaignAgent-SFT.svg?style=for-the-badge
[contributors-url]: https://github.com/ChaoyuWang04/AdCampaignAgent-SFT/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ChaoyuWang04/AdCampaignAgent-SFT.svg?style=for-the-badge
[forks-url]: https://github.com/ChaoyuWang04/AdCampaignAgent-SFT/network/members
[stars-shield]: https://img.shields.io/github/stars/ChaoyuWang04/AdCampaignAgent-SFT.svg?style=for-the-badge
[stars-url]: https://github.com/ChaoyuWang04/AdCampaignAgent-SFT/stargazers
[issues-shield]: https://img.shields.io/github/issues/ChaoyuWang04/AdCampaignAgent-SFT.svg?style=for-the-badge
[issues-url]: https://github.com/ChaoyuWang04/AdCampaignAgent-SFT/issues
[license-shield]: https://img.shields.io/github/license/ChaoyuWang04/AdCampaignAgent-SFT.svg?style=for-the-badge
[license-url]: https://github.com/ChaoyuWang04/AdCampaignAgent-SFT/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/samwang04/

[Python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[HuggingFace-badge]: https://img.shields.io/badge/🤗%20HuggingFace-FFD21E?style=for-the-badge
[HuggingFace-url]: https://huggingface.co/
[Pandas-badge]: https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/
[Matplotlib-badge]: https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white
[Matplotlib-url]: https://matplotlib.org/
[Seaborn-badge]: https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white
[Seaborn-url]: https://seaborn.pydata.org/
