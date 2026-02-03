import argparse
from rt.main import main
from rt.tasks import all_tasks, forecast_tasks

if __name__ == "__main__":

    load_ckpt_path = ""  # trained model path
    model_name = ""  # a short name for trained model
    assert (
        load_ckpt_path != ""
    ), "safety check to ensure paths are filled. skipping it will be similar to training from scratch."

    for leave_db in [
        "rel-amazon",
        "rel-hm",
        "rel-avito",
        "rel-trial",
        "rel-stack",
        "rel-f1",
    ]:
        print(f"Continued Pre-training without {leave_db}")

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
            max_local_ctx_len=1024,
            max_bfs_width=128,
            use_random_walk=False,
            use_random_sampling=False,
            use_connecting_nodes=False,
            num_walks=20000,
            walk_length=20,
            mask_prob=0.0,
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
