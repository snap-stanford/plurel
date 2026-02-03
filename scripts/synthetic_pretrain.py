import argparse
from rt.main import main
from rt.tasks import all_tasks, forecast_tasks, generate_rel_synthetic_tasks

if __name__ == "__main__":

    offset = 14000
    num_dbs = 2000
    num_test_dbs = 100
    for num_train_dbs in [8, 16, 32, 64, 128, 256, 512, 1024]:
        rel_synthetic_tasks = generate_rel_synthetic_tasks(
            offset=offset,
            num_dbs=num_dbs,
            num_train_dbs=num_train_dbs,
            num_test_dbs=num_test_dbs,
        )
        syn_autocomplete_clf_tasks = rel_synthetic_tasks["train_autocomplete_clf_tasks"]
        syn_autocomplete_reg_tasks = rel_synthetic_tasks["train_autocomplete_reg_tasks"]
        train_tasks = syn_autocomplete_clf_tasks + syn_autocomplete_reg_tasks
        eval_tasks = [
            t
            for t in forecast_tasks
            if t[0]
            in ["rel-hm", "rel-avito", "rel-stack", "rel-trial", "rel-f1", "rel-amazon"]
        ]
        eval_tasks += rel_synthetic_tasks["test_autocomplete_clf_tasks"]
        eval_tasks += rel_synthetic_tasks["test_autocomplete_reg_tasks"]

        max_steps_list = [4_001, 8_001, 16_001, 32_001, 64_001, 128_001, 256_001]
        for max_steps in max_steps_list:
            eval_freq = max_steps - 1
            main(
                # misc
                project="rt",
                eval_splits=["val", "test"],
                eval_freq=eval_freq,
                eval_pow2=False,
                max_eval_steps=80,
                load_ckpt_path=None,
                save_ckpt_dir=f"~/scratch/ckpts/syn-pt_d_seeds_{offset}_{num_dbs}_{num_train_dbs}_{num_test_dbs}_max_steps_{max_steps}",
                compile_=True,
                seed=0,
                # data
                train_tasks=train_tasks,
                eval_tasks=eval_tasks,
                batch_size=128,
                eval_batch_size=128,
                num_workers=2,
                ctx_len=1024,
                max_bfs_width=128,
                # optimization
                lr=5e-4,
                lr_schedule=True,
                wd=0.1,
                max_grad_norm=1.0,
                max_steps=max_steps,
                # model
                embedding_model="all-MiniLM-L12-v2",
                d_text=384,
                num_blocks=12,
                d_model=256,
                num_heads=8,
                d_ff=1024,
            )
