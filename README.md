# DietAI24

This repository contains the code for implementing DietAI24, an AI-based dietary assessment tool. The tool utilizes public food image datasets and implements various functionalities such as food code inference, portion size estimation, and nutrient amount estimation.

## Datasets

Two public food image datasets are used in this project:

- [ASA24](https://epi.grants.cancer.gov/asa24/resources/portionsize.html)
- [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k)

## Data Preparation

Data preparation details can be found in the following scripts:

- `asa_proc.py`: Prepares data from the ASA24 dataset.
- `nutrition5k_proc.py`: Prepares data from the Nutrition5k dataset.

## Implementation

The core functionalities of DietAI24 are implemented in the following scripts:

- `rag_food_code.py`: Implementation for food code inference.
- `rag_portion_size.py`: Implementation for portion size estimation.
- `nutrient_estimatte.py`: Implementation for nutrient amount estimation, evalution for ASA24.
- `nutrient_estimatte_mx.py`: Implementation for nutrient amount estimation, evaluation for Nutrition5k.

## Baseline Implementations

Implementations for our baselines can be found in the following scripts:

- `baseline_foodvisor.py`
- `baseline_calorieMaMa.py`
- `baseline_chatGPT.py`
- `baseline_ViT.py`

## Usage

1. Download the datasets from the provided links.
2. Prepare the data using the respective data preparation scripts.
3. Run the implementation scripts for food code inference, portion size estimation, and nutrient amount estimation.
4. Compare results with the baseline implementations.

## License

This project is licensed under the MIT License.

## Contact

For any inquiries, please contact .

