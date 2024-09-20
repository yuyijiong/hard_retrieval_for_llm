# Difficult retrieval tasks for long-context language models

[📜paper](https://arxiv.org/abs/2110.06767)

We find 2 types of difficult retrieval tasks for long-context language models (LLMs) :
1. multi-matching retrieval tasks
2. logic-based retrieval tasks

We construct a Key-Value pair Retrieval dataset and a student resumes analysis dataset to evaluate the performance of LLMs on these difficult retrieval tasks.

## Run
You can use ``test.py`` to test the performance of LLMs on these difficult retrieval tasks or other retrieval tasks.

You should modify the code in ``test.py`` to choose different tasks or models.

The tasks we provide are:
```python
{
"simple":simple retrieval,
"multi_step":multi-step retrieval,
"logic":logic based retrieval,
"multi_match":multi-matching retrieval
}
```

