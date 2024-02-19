
# Evaluation of Halucination Detection Methods for LLMs

```python
project/
│
├── data/
│   └── datasets.py
│
├── models/
│   └── llm_interaction.py
│
├── detection/
│   ├── method1.py
│   ├── method2.py
│   └── ...
│
├── metrics/
│   └── calculate_metrics.py
│
├── visualization/
│   └── visualize.py
│
└── main.py               # Main script to run the evaluation
```

## Notes regarding method implementatios

- **LBHD** (Logit-based Halucination Detection):
    - using "Instructing the Model" method to identify key concepts (for now), as no extra tool needed
    - not clear how the authors aggregated concept scores into sentence scores -> using same strategy for aggregation to sentence scores as for concept scores
    - using same llm for concept extraction as for response generation, to simulate real-world scenario

- **LM sv LM**:
    - prompts and inspiration from [github.com/maybenotime/PHD/LMvsLM_replicate](https://github.com/maybenotime/PHD/tree/main/LMvsLM_replicate)
    - 0 = true statement, 1 = hallucination
    - currently hardcoded max turns to 5, as in original paper