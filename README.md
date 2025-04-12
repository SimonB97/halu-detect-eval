
# Evaluation of Hallucination Detection Methods for LLMs

Implementation and evaluation framework for detecting hallucinations in Large Language Models.

## Methods Overview

| Method | Type | Description | Base Paper |
|--------|------|-------------|------------|
| LBHD | Token Probability | Logit-based analysis at concept/sentence level | [Varshney et al. 2023](https://doi.org/10.48550/arXiv.2307.03987) |
| SelfCheck | Consistency Check | Multi-sample comparison using NLI/BERTScore | [Manakul et al. 2023](https://doi.org/10.48550/arXiv.2303.08896) |
| LMvLM | Cross-Examination | Interactive verification (max 5 turns) | [Cohen et al. 2023](https://doi.org/10.48550/arXiv.2305.13281) |
| FLEEK | External Verification | Web-based fact verification | [Bayat et al. 2023](https://doi.org/10.48550/arXiv.2310.17119) |

## Implementation Details

### Method-Specific Modifications

| Method | Modification | Rationale |
|--------|--------------|-----------|
| LBHD | Model-based concept extraction | Avoid external NLP tools |
| FLEEK | Web-only evidence (10 results) | Simplified from original KG+web approach |
| SelfCheck | 5 samples @ temp=1.0 | Balance accuracy vs. compute |
| LMvLM | 5-turn limit | Match original paper design |

### Dataset Configuration

| Dataset | Size | Prompting | Ground Truth |
|---------|------|-----------|--------------|
| NQ_Open | 80 samples | 8-shot | GPT-4 verified |
| XSum | 80 samples | Direct | GPT-4 verified |

## Performance Results

### Effectiveness

| Method | AUC-ROC | AUC-PR | F1 | Pearson Corr. |
|--------|----------|---------|-----|---------------|
| SelfCheck-NLI | 0.79 | 0.58 | 0.60 | 0.66 |
| SelfCheck-BERT | 0.57 | 0.34 | 0.45 | 0.14 |
| LBHD (concept) | 0.69 | 0.42 | 0.51 | 0.51 |
| LBHD (sentence) | 0.61 | 0.41 | 0.46 | 0.27 |
| FLEEK | 0.52 | 0.29 | 0.26 | 0.28 |
| LMvLM | 0.57 | 0.31 | 0.37 | 0.10 |

### Efficiency

| Method | Runtime (80 samples) | GPU Required | Parallel Support |
|--------|---------------------|--------------|------------------|
| LBHD | 77.3s | No | Yes |
| SelfCheck-NLI | 143.5s | Yes | Yes |
| SelfCheck-BERT | 464.3s | Yes | Yes |
| FLEEK | 1000.8s | No | Limited by API |
| LMvLM | 1015.1s | No | Limited by API |

## Setup & Usage

### Requirements

```python
# Core dependencies
python>=3.8
torch>=2.0.0  # GPU recommended
transformers>=4.36.0
spacy>=3.7.0

# APIs (set as environment variables)
OPENAI_API_KEY="sk-..."        # For OpenAI models
TOGETHER_API_KEY="tok-..."     # For TogetherAI models
TAVILY_API_KEY="tvly-..."      # For FLEEK web search
```

### Running Experiments

The main entry point is `main.py`, which handles the complete evaluation pipeline. Configure the script through these flags:

```python
# Key Configuration Flags in main.py
PARALLEL = True       # Enable parallel processing for LLM calls
DEBUG = True         # Show API request details
DOWNLOAD_DATA = False # Set True to download fresh datasets
OVERWRITE = False    # Set True to overwrite existing result files
RANGE = 80          # Number of samples per dataset to process
```

Execute the complete pipeline:

```bash
# 1. Set required environment variables
export OPENAI_API_KEY="your-key"
export TOGETHER_API_KEY="your-key"  # Optional for alternative models
export TAVILY_API_KEY="your-key"    # Required for FLEEK

# 2. Run the evaluation
python main.py
```

The script will:
1. Load/download evaluation datasets (NQ_Open, XSum)
2. Generate LLM answers (with optional parallel processing)
3. Compute hallucination scores for selected methods
4. Generate ground truth labels using GPT-4
5. Save all results to the `results/` directory

### Output Files

Results are saved with model-specific filenames:
```
results/
├── {llm_name}_nqopen_with_answers__{model}.csv   # Model answers
├── {llm_name}_nqopen_with_scores__{model}.csv    # Detection scores
├── {llm_name}_nqopen_with_ground_truths__{model}.csv
├── {llm_name}_xsum_with_answers__{model}.csv
├── {llm_name}_xsum_with_scores__{model}.csv
├── {llm_name}_xsum_with_ground_truths__{model}.csv
└── {llm_name}_durations.csv                      # Runtime stats
```

## Citation

```bibtex
@thesis{benedict2024hallucination,
  title={Hallucination Detection in Large Language Models - Analysis and 
         Comparison of Various Approaches},
  author={Benedict, Simon},
  school={Technical University of Ulm},
  year={2024},
  type={Bachelor's Thesis}
}
```
