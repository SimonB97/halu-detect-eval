
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

- **LBHD**:
    - using "Instructing the Model" method to identify key concepts (for now), as no extra tool needed
    - not clear how the authors aggregated concept scores into sentence scores -> using same strategy for aggregation to sentence scores as for concept scores

