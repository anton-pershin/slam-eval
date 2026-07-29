# Spec: Multi-Score Support

## 1. Requirement Analysis

### Core Requirement

Scorers must be able to return both a scalar macro-score and multiple mini-scores in a single evaluation call, while maintaining backward compatibility and efficiency.

**Motivation**: 
- Some evaluations produce multiple metrics (e.g., Precision and Recall) that should all be stored
- LLM-as-a-judge scenarios require extracting multiple micro-scores from a single API call, then aggregating them into a macro-score
- Running multiple separate scorers would be wildly inefficient (N API calls for N scores)

### Expected Behavior

- **Simple scorers** (ExactMatch, IgnoreAllWhitespaces): Continue returning a single scalar score
- **Complex scorers** (IFBenchScorer, future LLM-as-judge): Return both an aggregated macro-score and a breakdown of mini-scores
- **Storage**: Preserve both the primary score (for backward compatibility) and mini-scores (for detailed analysis)

### Score Structure

Scorers return a `Score` object with:
- `primary: float` - the aggregated macro-score
- `sub_scores: dict[str, float] | None` - optional breakdown of mini-scores (None for simple scorers)

## 2. Tests

### 2.1 Scorer Interface Tests

1. **test_exact_match_returns_score**: Verify ExactMatch returns `Score(primary=1, sub_scores=None)` for matches and `Score(primary=0, sub_scores=None)` for non-matches
2. **test_ignore_all_whitespaces_returns_score**: Verify IgnoreAllWhitespaces returns `Score` objects with appropriate primary values
3. **test_ifbench_scorer_returns_score_with_sub_scores**: Verify IFBenchScorer returns `Score(primary=average, sub_scores={...per_checker_results...})`

### 2.2 Storage Adapter Tests

4. **test_save_with_score_objects**: Verify `save()` accepts `list[Score]` and correctly extracts primary values and sub_scores
5. **test_storage_format_with_sub_scores**: Verify JSONL output contains both `scores` and `sub_scores` fields with correct structure
6. **test_storage_format_without_sub_scores**: Verify JSONL output contains `sub_scores: [null, null, ...]` when scorers don't provide sub-scores
7. **test_load_returns_scores_and_sub_scores**: Verify `load()` returns result dicts with both `scores` and `sub_scores` fields

### 2.3 Integration Tests

8. **test_main_with_simple_scorer**: Verify main loop works with scorers returning `Score(primary=..., sub_scores=None)`
9. **test_main_with_complex_scorer**: Verify main loop works with scorers returning `Score` with sub_scores

## 3. Implementation Plan

### 3.1 Solution Design

#### Score Dataclass

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Score:
    primary: float
    sub_scores: dict[str, float] | None = None
```

**Rationale**: 
- Frozen dataclass ensures immutability
- Optional `sub_scores` allows simple scorers to ignore it
- Explicit separation of primary vs sub-scores

#### Scorer Interface

```python
class Scorer(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(self, y_true: Any, y_pred: Any) -> Score:
        ...
```

**Rationale**: Single return type simplifies type checking and storage handling.

#### Storage Adapter Changes

```python
def save(
    self,
    group_id: str,
    model: Model,
    eval_case_collection: EvalCaseCollection,
    scores: list[Score],  # Changed from list[int | float]
    model_answers: list[HasStr],
    **other_results,
) -> None:
    # ... existing code ...
    result_dict = {
        "group_id": group_id,
        "timestamp": datetime_now.timestamp(),
        "model": model.name,
        "eval_case_collection": eval_case_collection.name,
        "scores": [s.primary for s in scores],  # Extract primary values
        "sub_scores": [s.sub_scores for s in scores],  # Extract sub_scores
        "model_answers": model_answers,
    }
    # ... rest of method ...
```

**Rationale**: 
- Two separate fields maintain backward compatibility for `scores`
- Simple list comprehensions avoid complex serialization
- Parallel lists preserve per-case correspondence

#### Migration Strategy

1. **ExactMatch**: Return `Score(primary=int(match), sub_scores=None)`
2. **IgnoreAllWhitespaces**: Return `Score(primary=int(match), sub_scores=None)`
3. **IFBenchScorer**: Return `Score(primary=average, sub_scores={checker_name: result, ...})`

### 3.2 Todo List

1. [ ] Write tests described above
2. [ ] Run all tests and ensure they fail (TDD red phase)
3. [ ] Implement Score dataclass in `slam_eval/scorer.py`
4. [ ] Update Scorer base class `__call__` signature to return Score
5. [ ] Migrate ExactMatch, IgnoreAllWhitespaces, IFBenchScorer to return Score objects
6. [ ] Update EvalStorageAdapter.save() signature and implementation
7. [ ] Update main.py to handle Score objects (minimal changes)
8. [ ] Run all tests and ensure they pass (TDD green phase)
9. [ ] Run linters (black, isort, pylint, mypy)

### 3.3 Modification Summary

| File | Action |
|------|--------|
| `slam_eval/scorer.py` | Modified: Add Score dataclass, update Scorer.__call__ signature, migrate ExactMatch and IgnoreAllWhitespaces |
| `slam_eval/ifbench/scorer.py` | Modified: Migrate IFBenchScorer to return Score with sub_scores |
| `slam_eval/storage_adapter.py` | Modified: Update save() signature to accept list[Score], split into scores and sub_scores fields |
| `slam_eval/scripts/main.py` | Modified: Minor updates to handle Score objects (mostly transparent) |
| `tests/test_scorer.py` | Modified: Update tests to expect Score objects instead of scalars |
| `tests/test_storage_adapter.py` | Modified: Update tests to use Score objects and verify both scores and sub_scores fields |
| `tests/test_main.py` | Modified: Update assertions to check both scores and sub_scores in storage |
