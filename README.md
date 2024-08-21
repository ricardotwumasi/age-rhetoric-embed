# Age-Related Language Analysis in Workplace Contexts

This repository contains code for analyzing age-related language and potential biases in employer and union texts using Word Embedding Association Tests (WEAT).

## Overview

The code implements WEAT based on the methodology outlined in Caliskan et al. (2017) and applies it to the context of age-related language in the workplace. It builds upon the conceptual framework of "age work" introduced by Collien et al. (2016) and incorporates insights from Burn et al. (2022) on age stereotypes in job advertisements.

## Key Features

- Implementation of WEAT for measuring implicit associations in word embeddings
- Custom word sets for younger/older workers and positive/negative attributes
- Analysis of age-related language frequencies in employer and union texts

## Main Components

- `weat_test()`: Conducts the Word Embedding Association Test
- `get_embedding()`: Retrieves word embeddings (assumes pre-trained embeddings)
- Predefined word sets for target concepts and attributes

## Usage

1. Ensure you have the required dependencies installed (numpy, scipy, sklearn).
2. Load your pre-trained word embeddings into the `embeddings` dictionary.
3. Run the WEAT test using the provided function.

## References

- Burn, I., Button, P., Corella, L. M., & Neumark, D. (2022). Ageist Language in Job Ads and Age Discrimination in Hiring. Labour Economics, 77, 102019.
- Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334), 183-186. doi:10.1126/science.aal4230
- Collien, I., Sieben, B., & Müller-Camen, M. (2016). Age Work in Organizations: Maintaining and Disrupting Institutionalized Understandings of Higher Age. British Journal of Management, 27(4), 778-795. doi:10.1111/1467-8551.12198
- May, C., A. Wang, S. Bordia, S. R. Bowman and R. Rudinger (2019). 'On Measuring Social Biases in Sentence Encoders', arXiv.

  

## Authors

Max Lange;
Matt Flynn;
Ricardo Twumasi

Please direct all enquiries regarding the main manuscript or this code to: ricardo.twumasi@kcl.ac.uk; maximin.lange@kcl.ac.uk or matt.flynn@leicester.ac.uk
