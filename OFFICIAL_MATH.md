# Official MATH workload

The current official workload is a deterministic five-problem subset of the
Hendrycks MATH test set. It is intended to compare CPU timelines, not to claim
an accuracy score for the full 5,000-problem MATH test set.

- Source mirror: `EleutherAI/hendrycks_math`
- Revision: `21a5633873b6a120296cce3e2df9d5550074f4a3`
- Split: `test`
- Subjects: Algebra, Counting & Probability, Geometry, Number Theory,
  Prealgebra
- Selection: row 0 from each named subject

Prepare the pinned workload:

```bash
.venv/bin/pip install -r requirements-data.txt
.venv/bin/python scripts/data/prepare_official_math.py
```

Run the same three-repeat experiment with a requested 10 ms sample interval:

```bash
bash scripts/experiment/reproduce_official_math_10ms.sh
```

Each JSONL record contains its reference solution and complete source identity.
The experiment runner consumes the `id` and `question` fields; reference
solutions are retained for provenance and future accuracy evaluation.
