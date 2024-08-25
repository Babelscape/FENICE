
<div align="center">
  <img src="https://github.com/Babelscape/FENICE/blob/master/new_logo.png?raw=True" height="200", width="200">
</div>

# Factuality Evaluation of summarization based on Natural Language Inference and Claim Extraction

[![Conference](https://img.shields.io/badge/ACL-2024-4b44ce
)](https://2024.aclweb.org/)
[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](https://aclanthology.org/2024.findings-acl.841.pdf)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

<div align='center'>
  <img src="https://github.com/Babelscape/FENICE/blob/master/Sapienza_Babelscape.png?raw=True" height="70">
</div>
FENICE (Factuality Evaluation of Summarization based on Natural Language Inference and Claim Extraction) is a factuality-oriented metric for summarization. 
This package implements the FENICE metric, allowing users to evaluate the factual consistency of document summaries.

## Overview

Factual consistency in summarization is critical for ensuring that the generated summaries accurately reflect the content of the original documents. 
FENICE leverages NLI and claim extraction techniques to assess the factual alignment between a summary and its corresponding document.

For more details, you can read the full paper: [FENICE: Factuality Evaluation of Summarization based on Natural Language Inference and Claim Extraction](https://aclanthology.org/2024.findings-acl.841/).

## 🛠️ Installation

Create a conda environment:
```sh
conda create -n FENICE python=[PYTHON_VERSION]
conda activate FENICE
```

To install the FENICE package, you can use `pip`:

```sh
pip install FENICE
```

install spacy model:
```sh
python -m spacy download en_core_web_sm
```

## Requirements

The package requires the following dependencies:

	•	spacy==3.7.4
	•	fastcoref==2.1.6
	•	transformers~=4.38.2
	•	sentencepiece==0.2.0
	•	scikit-learn==1.5.0

These will be installed automatically when you install the FENICE package.

## Usage

Here’s how you can use the FENICE package to evaluate the factual consistency of document summaries:

```sh
from metric.FENICE import FENICE
fenice = FENICE()



document = '''
Simone Biles’ Olympic return is off to a sparkling start at Paris 2024 as the Americans competed in women’s qualifying Sunday (28 July). The U.S. is well in front with a total team score of 172.296, followed by Italy some 5.435 points back at 166.861. The People’s Republic of China is third with a 166.628.

Reigning world silver medallists Brazil competed in the day’s final subdivision and sit fourth (166.499). In the all-around, Biles, the 2016 gold medallist, scored 59.566 ahead of 2022 world all-around champion Rebeca Andrade (57.700). Reigning champ Suni Lee was third, posting a 56.132. Jordan Chiles earned the fourth highest score at 56.065 but won’t advance to Thursday’s (1 August) all-around final due to two-per-country restrictions. Algeria’s Kaylia Nemour rounds out the top five.

Three years ago, the American withdrew from the women’s team final and four subsequent individual finals at Tokyo 2020 to prioritize her mental health as she dealt with the ‘twisties.’ That seemed like a distant memory Sunday.

Biles, 27, entered Bercy Arena to massive applause, looking relaxed as she smiled and waved to the audience. She looked even more relaxed on the balance beam where in the span of some 79 seconds, she put on a clinic, executing a near flawless routine that included a two layout stepout series and a full-twisting double back dismount. Biles earned a 14.733 for the routine.

In the warm-up for the second rotation on the floor exercise, Biles appeared to tweek her left ankle on her Biles I (double layout half out). When she took to the mat for her competitive routine, her ankle was heavily taped. She delivered a solid, if not bouncy, routine on the event for a 14.666.

As she came off the podium, coach Cecile Landi, a 1996 Olympian for France, asked if she was OK. Biles confirmed she was. But the uncertainty continued through the vault warm-up where at one point she crawled nearly two-thirds of the way back to the starting position before hopping on her right leg.

Later, Biles could be seen waving to her parents, Ron and Nellie, as well as sharing a laugh and several smiles with Landi. When it came time for competition, there was no hint of an issue as she boomed her trademark Yurchenko double pike to the rafters, needing several steps backward to control it. She earned a massive 15.800.

“She felt a little something in her calf. That’s all,” Landi told reporters afterward, adding that Biles was not thinking of leaving the competition. “Never in her mind.” The injury, Landi explained, had popped up a few weeks ago but had subsided in the training leading to Paris. “It felt better at the end [of competition today],” she said later. “On bars, it started to feel better.”

Biles closed out her spectacular on the uneven bars with a hit set and a 14.433, the relief pouring out through her megawatt smile. She embraced coach Laurent Landi before stopping near a scoreboard to soak in the moment. The crowd roared, acknowledging her spectacular return. Before leaving the podium, she blew kisses and waved to her adoring fans.

The American is now set for five of six medal rounds, advancing with the team, in the all-around, and on the vault, beam and floor exercise. “It was pretty amazing. 59.5, and four-for-four. Not perfect,” Landi assessed her pupil’s performance. “She still can improve even.”
'''

summary = 'Simone Biles made a triumphant return to the Olympic stage at the Paris 2024 Games, competing in the women’s gymnastics qualifications. Overcoming a previous struggle with the “twisties” that led to her withdrawal from events at the Tokyo 2020 Olympics, Biles dazzled with strong performances on all apparatus, helping the U.S. team secure a commanding lead in the qualifications. Her routines, including a near-flawless balance beam performance and a powerful Yurchenko double pike vault, showcased her resilience and skill, drawing enthusiastic support from a star-studded audience'

batch = [
    {"document": document, "summary": summary}
]

results = fenice.score_batch(batch)
print(results)

[{'score': 0.8484095427307433, 'alignments': [{'score': 0.9968091815244406, 'summary_claim': 'Simone Biles made a triumphant return to the Olympic stage at the Paris 2024 Games.', 'source_passage': '\n        Simone Biles’ Olympic return is off to a sparkling start at Paris 2024 as the Americans competed in women’s qualifying Sunday (28 July). The U.S. is well in front with a total team score of 172.296, followed by Italy some 5.435 points back at 166.861. The People’s Republic of China is third with a 166.628.\n    \n         Reigning world silver medallists Brazil competed in the day’s final subdivision and sit fourth (166.499). In the all-around, Biles, the 2016 gold medallist, scored 59.566 ahead of 2022 world all-around champion Rebeca Andrade (57.700).'}, {'score': 0.9985068442765623, 'summary_claim': 'Biles competed in the women’s gymnastics qualifications.', 'source_passage': '\n        Simone Biles’ Olympic return is off to a sparkling start at Paris 2024 as the Americans competed in women’s qualifying Sunday (28 July). The U.S. is well in front with a total team score of 172.296, followed by Italy some 5.435 points back at 166.861. The People’s Republic of China is third with a 166.628.\n    \n         Reigning world silver medallists Brazil competed in the day’s final subdivision and sit fourth (166.499). In the all-around, Biles, the 2016 gold medallist, scored 59.566 ahead of 2022 world all-around champion Rebeca Andrade (57.700).'}, {'score': 0.9983009036513977, 'summary_claim': "Biles overcame a previous struggle with the 'twisties' that led to her withdrawal from events at the Tokyo 2020 Olympics.", 'source_passage': 'Three years ago, the American withdrew from the women’s team final and four subsequent individual finals at Tokyo 2020 to prioritize her mental health as she dealt with the ‘twisties.’ That seemed like a distant memory Sunday.\n    \n         Biles, 27, entered Bercy Arena to massive applause, looking relaxed as she smiled and waved to the audience. She looked even more relaxed on the balance beam where in the span of some 79 seconds, she put on a clinic, executing a near flawless routine that included a two layout stepout series and a full-twisting double back dismount. Biles earned a 14.733 for the routine.\n    \n        '}, {'score': 0.9821975510567427, 'summary_claim': 'Biles dazzled with strong performances on all apparatus.', 'source_passage': 'DOCUMENT'}, {'score': 0.9991946243681014, 'summary_claim': 'The U.S. team secured a commanding lead in the qualifications.', 'source_passage': '\n        Simone Biles’ Olympic return is off to a sparkling start at Paris 2024 as the Americans competed in women’s qualifying Sunday (28 July). The U.S. is well in front with a total team score of 172.296, followed by Italy some 5.435 points back at 166.861. The People’s Republic of China is third with a 166.628.\n    \n         Reigning world silver medallists Brazil competed in the day’s final subdivision and sit fourth (166.499). In the all-around, Biles, the 2016 gold medallist, scored 59.566 ahead of 2022 world all-around champion Rebeca Andrade (57.700).'}, {'score': 0.9942512132693082, 'summary_claim': 'Her routines showcased her resilience and skill.', 'source_passage': 'DOCUMENT'}, {'score': -0.03039351903134957, 'summary_claim': 'Her routines drew enthusiastic support from a star-studded audience.', 'source_passage': 'Three years ago, the American withdrew from the women’s team final and four subsequent individual finals at Tokyo 2020 to prioritize her mental health as she dealt with the ‘twisties.’'}]}]

```
## Claim Extraction
Our t5-based claim extractor is now available on [🤗 Hugging Face](https://huggingface.co/Babelscape/t5-base-summarization-claim-extractor)!

## Long-Form summarization Evaluation
Check-out our dataset of annotations for long-form summarization evaluation (Section 4.4 in the paper):

🤗 [Babelscape/story-summeval](https://huggingface.co/datasets/Babelscape/story-summeval)

We provide binary factuality annotations for summaries of stories (from Gutenberg and Wikisource).
In total, our dataset features 319 (story, summary) pairs.



<!-- ## AggreFact evaluation

To replicate the evaluation of FENICE on AggreFact:
```sh
PYTHONPATH=. python eval/aggrefact.py
```
This script utilizes the data from the [AggreFact repository](https://github.com/Liyan06/AggreFact/tree/main/data), which we downloaded and converted into .jsonl format.  -->


## License
This work is under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
## Citation Information


```bibtex

@inproceedings{scire-etal-2024-fenice,
    title = "{FENICE}: Factuality Evaluation of summarization based on Natural language Inference and Claim Extraction",
    author = "Scir{\`e}, Alessandro and Ghonim, Karim and Navigli, Roberto",
    editor = "Ku, Lun-Wei  and Martins, Andre and Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.841",
    pages = "14148--14161",
}
```
