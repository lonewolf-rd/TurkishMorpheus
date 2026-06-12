import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List, Callable, Tuple


BASE = Path(__file__).resolve().parent.parent


def _banner(title: str, ch: str = "="):
    line = ch * 78
    print(f"\n{line}\n  {title}\n{line}", flush=True)


def _check_file(path: Path, label: str) -> bool:
    exists = path.exists()
    status = "OK" if exists else "MISSING"
    size = f"({path.stat().st_size / 1e6:.1f} MB)" if exists else ""
    print(f"   [{status:7s}] {label:<28s} {path}  {size}")
    return exists


def stage_data() -> bool:
    _banner("STAGE 1/6  ·  data preprocess + analyze")
    from src.model_development.data.preprocessor import DataPreprocessor
    from src.model_development.data.analyzer import DatasetAnalyzer

    raw_path = BASE / "src/model_development/artifacts/datasets/raw/corpus.txt"
    train_path = BASE / "src/model_development/artifacts/datasets/splits/train.txt"
    test_path = BASE / "src/model_development/artifacts/datasets/splits/test.txt"

    if train_path.exists() and test_path.exists():
        print(f"   train/test splits already exist — skipping data ingest.")
        _check_file(train_path, "train.txt")
        _check_file(test_path, "test.txt")
        return True

    if not raw_path.exists():
        print("   raw corpus missing — ingesting from configured local_corpus_path...")
        DataPreprocessor().process_pipeline()
    else:
        print(f"   raw corpus exists at {raw_path}, skipping ingest")

    print("\n   running DatasetAnalyzer.analyze_and_split() ...")
    DatasetAnalyzer().analyze_and_split()

    return _check_file(train_path, "train.txt") and _check_file(test_path, "test.txt")


def stage_benchmark() -> bool:
    _banner("STAGE 2/6  ·  classical tokenizer benchmark (BPE/ByteBPE/Unigram/WordPiece + Morfessor)")
    from src.benchmarker.benchmarks.classical import TokenizerBenchmarker

    morfessor_path = BASE / "src/model_development/artifacts/tokenizers/classical/morfessor_model.bin"

    hard_words = [
        "evlerimizdekiler", "muvaffakiyetsizleştiriciler",
        "gidebileceklerindenmişsiniz", "anlaşılamamaktadır",
        "kitap", "ev", "geliyorum",
    ]

    benchmarker = TokenizerBenchmarker(test_sentences=hard_words)
    benchmarker.run_benchmark()
    return _check_file(morfessor_path, "morfessor_model.bin")


def stage_dataset() -> bool:
    _banner("STAGE 3/6  ·  build sentence cache for Morpheus training")
    from src.model_development.training.dataset import build_sentence_cache

    train_txt = str(BASE / "src/model_development/artifacts/datasets/splits/train.txt")
    test_txt = str(BASE / "src/model_development/artifacts/datasets/splits/test.txt")
    morfessor_path = str(BASE / "src/model_development/artifacts/tokenizers/classical/morfessor_model.bin")
    word_vocab_path = str(BASE / "src/model_development/artifacts/datasets/splits/word_vocab.pt")
    root_vocab_path = str(BASE / "src/model_development/artifacts/datasets/splits/root_vocab.pt")
    train_cache = str(BASE / "src/model_development/artifacts/datasets/splits/train_sentences.pt")
    test_cache = str(BASE / "src/model_development/artifacts/datasets/splits/test_sentences.pt")

    print("   building train sentence cache...")
    build_sentence_cache(
        txt_path=train_txt,
        cache_path=train_cache,
        word_vocab_path=word_vocab_path,
        root_vocab_path=root_vocab_path,
        morfessor_path=morfessor_path,
        max_sentences=5_000_000,
        max_sent_len=32,
        word_vocab_top_k=120_000,
        word_vocab_min_freq=5,
        root_vocab_top_k=30_000,
        root_vocab_min_freq=2,
    )

    print("\n   building test sentence cache...")
    build_sentence_cache(
        txt_path=test_txt,
        cache_path=test_cache,
        word_vocab_path=word_vocab_path,
        root_vocab_path=root_vocab_path,
        morfessor_path=morfessor_path,
        max_sentences=500_000,
        max_sent_len=32,
        word_vocab_top_k=120_000,
        word_vocab_min_freq=5,
        root_vocab_top_k=30_000,
        root_vocab_min_freq=2,
    )

    ok = True
    ok &= _check_file(Path(train_cache), "train_sentences.pt")
    ok &= _check_file(Path(test_cache), "test_sentences.pt")
    ok &= _check_file(Path(word_vocab_path), "word_vocab.pt")
    ok &= _check_file(Path(root_vocab_path), "root_vocab.pt")
    return ok


def stage_train() -> bool:
    _banner("STAGE 4/6  ·  Morpheus training")
    from src.model_development.training.trainer import MorpheusTrainer, TrainingConfig

    checkpoint_dir = BASE / "src/model_development/artifacts/checkpoints"
    train_cache = str(BASE / "src/model_development/artifacts/datasets/splits/train_sentences.pt")
    test_cache = str(BASE / "src/model_development/artifacts/datasets/splits/test_sentences.pt")
    word_vocab_path = str(BASE / "src/model_development/artifacts/datasets/splits/word_vocab.pt")

    config = TrainingConfig(
        train_cache_path=train_cache,
        val_cache_path=test_cache,
        word_vocab_path=word_vocab_path,
        checkpoint_dir=str(checkpoint_dir),
        run_name="turkish_morpheus_a100_v3",
    )

    trainer = MorpheusTrainer(config, use_wandb=True)
    trainer.train()

    final_ckpt = checkpoint_dir / f"{config.run_name}_final.pt"
    return _check_file(final_ckpt, "final checkpoint")


def stage_tokenizer() -> bool:
    _banner("STAGE 5/6  ·  build MorpheusTokenizer (50K vocab)")
    import torch
    from src.model_development.model.morpheus import Morpheus
    from src.model_development.tokenization.morpheus_tokenizer import (
        MorpheusTokenizer,
        build_morpheus_vocab,
    )

    checkpoint_path = BASE / "src/model_development/artifacts/checkpoints/turkish_morpheus_a100_v3_final.pt"
    if not checkpoint_path.exists():
        for cand in [
            "turkish_morpheus_a100_v3_best.pt",
            "turkish_morpheus_a100_release_best.pt",
            "turkish_morpheus_a100_best.pt",
        ]:
            fallback = BASE / f"src/model_development/artifacts/checkpoints/{cand}"
            if fallback.exists():
                checkpoint_path = fallback
                break
        else:
            print(f"   no checkpoint found — train first.")
            return False

    corpus_path = str(BASE / "src/model_development/artifacts/datasets/splits/train.txt")
    output_dir = BASE / "src/model_development/artifacts/tokenizers/morpheus_50k"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   loading checkpoint: {checkpoint_path.name}  (device={device})")

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = Morpheus(
        char_dim=cfg.char_dim,
        char_embed_dim=cfg.char_embed_dim,
        case_embed_dim=cfg.case_embed_dim,
        n_layers_encoder=cfg.n_layers_encoder,
        n_layers_detector=cfg.n_layers_detector,
        num_heads=cfg.num_heads,
        max_word_len=cfg.max_word_len,
        max_segs=cfg.max_segs,
        dropout=cfg.dropout,
        threshold=cfg.threshold,
        pos_weight=cfg.pos_weight,
        count_loss_w=getattr(cfg, "count_loss_w", 0.3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    print("   building 50K vocab from corpus...")
    vocab = build_morpheus_vocab(
        morpheus_model=model,
        corpus_path=corpus_path,
        vocab_size=50_000,
        min_freq=2,
        device=device,
    )

    tokenizer = MorpheusTokenizer(
        morpheus_model=model,
        vocab=vocab,
        device=device,
    )
    tokenizer.save(output_dir)

    print(f"\n   vocab stats: {tokenizer.vocab_stats()}")
    return _check_file(output_dir / "vocab.json", "vocab.json") and _check_file(
        output_dir / "tokenizer_config.json", "tokenizer_config.json"
    )


def stage_eval() -> bool:
    _banner("STAGE 6/6  ·  evaluations (paper_evaluation + sigmorphon + trmmlu + morphscore)")

    print("\n   [6a] running paper_evaluation orchestrator...")
    from src.benchmarker.benchmarks.paper import PaperEvaluator

    checkpoint_path = BASE / "src/model_development/artifacts/checkpoints/turkish_morpheus_a100_v3_final.pt"
    if not checkpoint_path.exists():
        for cand in [
            "turkish_morpheus_a100_v3_best.pt",
            "turkish_morpheus_a100_release_best.pt",
            "turkish_morpheus_a100_best.pt",
        ]:
            fallback = BASE / f"src/model_development/artifacts/checkpoints/{cand}"
            if fallback.exists():
                checkpoint_path = fallback
                break
        else:
            print(f"   no checkpoint found — train first.")
            return False

    tokenizer_dir = BASE / "src/model_development/artifacts/tokenizers/morpheus_50k"
    morfessor_path = BASE / "src/model_development/artifacts/tokenizers/classical/morfessor_model.bin"
    train_corpus = BASE / "src/model_development/artifacts/datasets/splits/train.txt"
    test_corpus = BASE / "src/model_development/artifacts/datasets/splits/test.txt"
    word_vocab_path = BASE / "src/model_development/artifacts/datasets/splits/word_vocab.pt"
    benchmarker_results = BASE / "src/model_development/artifacts/tokenizers/classical"
    paper_eval_dir = BASE / "src/benchmarker/results/paper_eval"

    evaluator = PaperEvaluator(
        checkpoint_path=str(checkpoint_path),
        tokenizer_dir=str(tokenizer_dir) if tokenizer_dir.exists() else None,
        morfessor_path=str(morfessor_path),
        train_corpus_path=str(train_corpus),
        test_corpus_path=str(test_corpus),
        word_vocab_path=str(word_vocab_path),
        benchmarker_results_dir=str(benchmarker_results),
        preferred_classical_vocab=64000,
        output_dir=str(paper_eval_dir),
        seed=1337,
    )
    evaluator.run_all()

    sigmorphon_gold = BASE / "data/sigmorphon_tr/tur.gold"
    if sigmorphon_gold.exists():
        print("\n   [6b] running sigmorphon_eval against tur.gold...")
        from src.benchmarker.benchmarks.sigmorphon import (
            load_sigmorphon_inflection_gold,
            build_segmenters,
            evaluate,
        )

        entries = load_sigmorphon_inflection_gold(str(sigmorphon_gold))
        if entries:
            segmenters = build_segmenters(
                checkpoint_path=str(checkpoint_path),
                morfessor_path=str(morfessor_path),
                benchmarker_results_dir=str(benchmarker_results),
            )
            evaluate(entries, segmenters, str(paper_eval_dir / "sigmorphon"))
    else:
        print(f"\n   [6b] SIGMORPHON gold not found at {sigmorphon_gold} — skipping")

    print("\n   [6c] running TR-MMLU %TR / %Pure / Rényi evaluation...")
    try:
        from src.benchmarker.benchmarks.trmmlu_eval import run as trmmlu_run
        trmmlu_run(classical_vocab=64000, use_analyzer=True)
    except Exception:
        print("   [6c] trmmlu_eval FAILED (non-fatal):")
        traceback.print_exc()

    print("\n   [6d] running MorphScore v2 evaluation (UD_Turkish-Kenet gold)...")
    try:
        from src.benchmarker.benchmarks.morphscore_eval import run as morphscore_run
        morphscore_run(classical_vocab=64000)
    except Exception:
        print("   [6d] morphscore_eval FAILED (non-fatal):")
        traceback.print_exc()

    print("\n   [6e] running roundtrip reconstruction evaluation...")
    try:
        from src.benchmarker.benchmarks.roundtrip_eval import run as roundtrip_run
        roundtrip_run(classical_vocab=64000)
    except Exception:
        print("   [6e] roundtrip_eval FAILED (non-fatal):")
        traceback.print_exc()

    print("\n   [6f] generating comparison figures...")
    try:
        from src.benchmarker.visualization.eval_report import run as eval_report_run
        eval_report_run(base=BASE)
    except Exception:
        print("   [6f] eval_report FAILED (non-fatal):")
        traceback.print_exc()

    return True


STAGES: List[Tuple[str, Callable[[], bool]]] = [
    ("data", stage_data),
    ("benchmark", stage_benchmark),
    ("dataset", stage_dataset),
    ("train", stage_train),
    ("tokenizer", stage_tokenizer),
    ("eval", stage_eval),
]


def run_stages(stage_names: List[str]):
    stage_map = dict(STAGES)
    ordered = [s for s, _ in STAGES if s in stage_names]
    unknown = [s for s in stage_names if s not in stage_map]
    if unknown:
        print(f"Unknown stage(s): {unknown}. Valid: {[s for s, _ in STAGES]}")
        sys.exit(1)

    t_start = time.time()
    results = []
    for name in ordered:
        fn = stage_map[name]
        t0 = time.time()
        try:
            ok = fn()
        except Exception as e:
            print(f"\n[run_pipeline] STAGE '{name}' RAISED EXCEPTION:")
            traceback.print_exc()
            ok = False
        elapsed = time.time() - t0
        results.append((name, ok, elapsed))
        if not ok:
            print(f"\n[run_pipeline] stage '{name}' did not complete cleanly. Stopping.")
            break

    _banner("PIPELINE SUMMARY")
    total = time.time() - t_start
    for name, ok, elapsed in results:
        mark = "OK" if ok else "FAIL"
        m, s = divmod(int(elapsed), 60)
        print(f"   [{mark:4s}] {name:<12s} {m:>3d}m{s:02d}s")
    print(f"\n   TOTAL: {int(total // 3600)}h{int((total % 3600) // 60):02d}m")


def main():
    parser = argparse.ArgumentParser(
        prog="src.run_pipeline",
        description="Orchestrate Morpheus pipeline stages",
    )
    parser.add_argument(
        "--stage",
        choices=["all"] + [s for s, _ in STAGES],
        default="all",
        help="Single stage to run, or 'all' for full pipeline",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        help="Multiple stages (overrides --stage)",
    )
    parser.add_argument(
        "--from",
        dest="from_stage",
        choices=[s for s, _ in STAGES],
        help="Run from this stage onward",
    )

    args = parser.parse_args()
    all_stage_names = [s for s, _ in STAGES]

    if args.stages:
        chosen = args.stages
    elif args.from_stage:
        start_idx = all_stage_names.index(args.from_stage)
        chosen = all_stage_names[start_idx:]
    elif args.stage == "all":
        chosen = all_stage_names
    else:
        chosen = [args.stage]

    print(f"[run_pipeline] running stages: {chosen}")
    run_stages(chosen)


if __name__ == "__main__":
    main()
