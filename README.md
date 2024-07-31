# FENICE

FENICE (Factuality Evaluation of Summarization based on Natural Language Inference and Claim Extraction) is a factuality-oriented metric for summarization. 
This package implements the FENICE metric, allowing users to evaluate the factual consistency of document summaries.

## Overview

Factual consistency in summarization is critical for ensuring that the generated summaries accurately reflect the content of the original documents. 
FENICE leverages NLI and claim extraction techniques to assess the factual alignment between a summary and its corresponding document.

For more details, you can read the full paper: [Factuality Evaluation of Summarization based on Natural Language Inference and Claim Extraction](https://arxiv.org/abs/2403.02270).

## Installation

To install the FENICE package, you can use `pip`:

```sh
pip install git+https://github.com/Babelscape/FENICE.git
```

## Requirements

The package requires the following dependencies:

	•	spacy==3.7.4
	•	en_core_web_sm
	•	fastcoref==2.1.6
	•	transformers~=4.38.2
	•	sentencepiece==0.2.0
	•	scikit-learn==1.5.0

These will be installed automatically when you install the FENICE package.

Usage

Here’s how you can use the FENICE package to evaluate the factual consistency of document summaries:

Step 1: Import the package

```sh
from FENICE import FENICE
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
    {"document": document "summary": "Test summary."}
]

results = fenice.score_batch(batch)
print(results)
```
To cite this work in your research:
```sh
@misc{scirè2024fenicefactualityevaluationsummarization,
      title={FENICE: Factuality Evaluation of summarization based on Natural language Inference and Claim Extraction}, 
      author={Alessandro Scirè and Karim Ghonim and Roberto Navigli},
      year={2024},
      eprint={2403.02270},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2403.02270}, 
}
```

## License
This work is under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

