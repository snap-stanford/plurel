import argparse
from rt.main import main
from rt.tasks import all_tasks, forecast_tasks

if __name__ == "__main__":

    load_ckpt_path = "~/scratch/rt_hf_ckpts/synthetic-pretrain_rdb_1024_size_4b.pt"
    model_name = "synthetic-pretrain_rdb_1024_size_4b"
    for leave_db in [
        "rel-amazon",
        "rel-hm",
        "rel-avito",
        "rel-trial",
        "rel-stack",
        "rel-f1",
    ]:
        print(f"Continued Pre-training of {model_name} without {leave_db}")

        main(
            # misc
            project="rt",
            eval_splits=["val", "test"],
            eval_freq=2000,
            eval_pow2=False,
            max_eval_steps=80,
            load_ckpt_path=load_ckpt_path,
            save_ckpt_dir=f"~/scratch/ckpts/baselines/cpt/{model_name}/leave_{leave_db}",
            compile_=True,
            seed=0,
            # data
            train_tasks=[t for t in all_tasks if t[0] != leave_db],
            eval_tasks=[t for t in forecast_tasks if t[0] == leave_db],
            batch_size=128,
            eval_batch_size=128,
            num_workers=2,
            ctx_len=1024,
            max_bfs_width=128,
            # optimization
            lr=5e-4,
            wd=0.1,
            lr_schedule=True,
            max_grad_norm=1.0,
            max_steps=50_001,
            # model
            embedding_model="all-MiniLM-L12-v2",
            d_text=384,
            num_blocks=12,
            d_model=256,
            num_heads=8,
            d_ff=1024,
        )
