"""Old path -> current path for every file that moved during the campaign. The report builder
(report/scripts/stage2.py) uses it to rewrite file citations in the report text and the experiment register."""
MOVES = {
    # documentation
    "CAMPAIGN_REPORT.md": "docs/CAMPAIGN_REPORT.md",
    "TEST_AND_SUBMIT_FINETUNED.md": "docs/TEST_AND_SUBMIT_FINETUNED.md",
    "lab/KAGGLE_RUNS_26SEP.md": "docs/KAGGLE_RUNS.md",
    "lab/EXPERIMENTS_27SEP.md": "docs/FOLLOW_UP_EXPERIMENTS.md",
    "docs/notes/STATE.md": "docs/notes/STATE.md",
    "lab/TOP10_PLAN.md": "docs/notes/TOP10_PLAN.md",
    "lab/QUERY_v2.md": "docs/notes/QUERY_v2.md",
    "lab/cleanup_deleted_26sep.txt": "docs/notes/cleanup_deleted_26sep.txt",
    # raw logs as received (renamed without dates)
    "results/live_logs/live_v17_25sep.txt": "results/live_logs/live_v17.txt",
    "results/live_logs/kaggle_run1_exp1_26sep.txt": "results/live_logs/kaggle_run1_exp1.txt",
    "results/live_logs/kaggle_run2_resid_26sep.txt": "results/live_logs/kaggle_run2_resid.txt",
    "results/live_logs/live_exp1_sf_26sep.txt": "results/live_logs/live_exp1_sf.txt",
    # result files (JSON) written/read by the root scoring scripts
    "holdout_v12_3_stock_scores.json": "results/holdout_v12_3_stock_scores.json",
    "holdout_v12_3_simcno_frozen_scores.json": "results/holdout_v12_3_simcno_frozen_scores.json",
    "boosted_scores.json": "results/boosted_scores.json",
    "finetuned_scores.json": "results/finetuned_scores.json",
    "eda_profile.json": "results/eda_profile.json",
    "generalisation_map.json": "results/generalisation_map.json",
    # raw console logs
    "bnfc_dryrun_out.txt": "results/logs/bnfc_dryrun_out.txt",
    "combo_symbolic_out.txt": "results/logs/combo_symbolic_out.txt",
    "compare_v11_v12_out.txt": "results/logs/compare_v11_v12_out.txt",
    "gate3_aoa15_out.txt": "results/logs/gate3_aoa15_out.txt",
    "linear_gate_out.txt": "results/logs/linear_gate_out.txt",
    "preflight_v10_out.txt": "results/logs/preflight_v10_out.txt",
    "preflight_v11_out.txt": "results/logs/preflight_v11_out.txt",
    "preflight_v12_out.txt": "results/logs/preflight_v12_out.txt",
    "preflight_v13_out.txt": "results/logs/preflight_v13_out.txt",
    "v10_live.txt": "results/logs/v10_live.txt",
    # tools and registers
    "lab/jev_gate.py": "tools/jev_gate.py",
    "closed_directions.json": "results/closed_directions.json",
}

# Final layout: scripts that used to sit at the repository root, cited by bare name in the report and the register.
GROUPS = {
    "src/training": "finetune_fno_kaggle finetune_bnfc_kaggle finetune_cno_kaggle boost_operator_kaggle "
                    "cluster_specialist_kaggle check_kaggle_uploads spectral_augment spectral_gain swa_average",
    "src/evaluation": "preflight score_holdout_v12_3 score_finetuned_ckpt score_boosted_ckpt score_cno_ckpt score_unet_ckpt "
                      "score_one_trajectory build_fno_zip build_cno_zip _vendor_requests_stub rpdebench_unet",
    "src/splits": "make_matched_splits build_holdout_v12_3 build_live_pool window_similarity_aoa15 window_similarity_select",
    "experiments/early_diagnostics": "generalisation_map holdout_hardness_distribution local_ref_aoa15 local_ref_diag "
                                     "linear_micro_gate linear_micro_gate_aoa15 combo_forms combo_symbolic compare_v11_v12 "
                                     "adabn_test wiseft_test pilot_dpot_test pilot_unet_select",
}
for _d, _names in GROUPS.items():
    for _n in _names.split():
        MOVES[_n + ".py"] = f"{_d}/{_n}.py"
for _s in ("holdout_v12_3.json", "train_pool_v12_3.json", "train_pool_all81.json", "matched_splits.json", "live_pool.json"):
    MOVES[_s] = f"data/splits/{_s}"
