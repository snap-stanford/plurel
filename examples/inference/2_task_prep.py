"""Step 2 — Define the prediction task.

Reads the task from ``config.py`` — the entity, the timestamp, the target, and the
labeled train/val/test rows — and writes the task tables next to the database from
step 1. This is what turns your data into a supervised prediction problem.

    pixi run python examples/inference/2_task_prep.py
"""

import config
import pipeline


def main():
    if not (pipeline.dataset_dir(config.DB_NAME) / "db").exists():
        raise SystemExit("Run 1_data_prep.py first — no RelBench database found on disk.")

    task = config.TASK
    split_tables = pipeline.build_task_tables(task)
    out = pipeline.save_task_tables(config.DB_NAME, task["name"], split_tables, task_cfg=task)

    print(f"[step 2] task '{task['name']}' ({task['task_type']})")
    print(f"   predict '{task['target_col']}' for '{task['entity_table']}' at '{task['time_col']}'")
    for split, table in split_tables.items():
        print(f"   - {split}: {len(table.df)} labeled rows")
    print(f"[step 2] wrote task tables -> {out}")
    print("\nNext: run inference with 3_predict.py")


if __name__ == "__main__":
    main()
