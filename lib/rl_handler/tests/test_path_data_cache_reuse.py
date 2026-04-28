from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

RL_HANDLER_ROOT = Path(__file__).resolve().parents[1]
if str(RL_HANDLER_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_HANDLER_ROOT))

_MAIN_SPEC = importlib.util.spec_from_file_location(
    "rl_handler_main_for_tests",
    RL_HANDLER_ROOT / "__main__.py",
)
if _MAIN_SPEC is None or _MAIN_SPEC.loader is None:
    raise RuntimeError("Failed to load rl_handler __main__.py for tests.")
RL_MAIN = importlib.util.module_from_spec(_MAIN_SPEC)
sys.modules[_MAIN_SPEC.name] = RL_MAIN
_MAIN_SPEC.loader.exec_module(RL_MAIN)


class TestPathDataCacheReuse(unittest.TestCase):
    def test_resolve_processed_data_dir_accepts_explicit_namespace_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td) / "save_data"
            namespace_dir = Path(td) / "processed_data" / "onnx_frontier_size_64_shared"
            args = argparse.Namespace(path_data=namespace_dir.as_posix())

            resolved = RL_MAIN._resolve_processed_data_dir(
                args=args,
                data_root=data_root,
                cache_namespace="onnx_frontier_size_64_shared",
            )

            self.assertEqual(resolved, namespace_dir)
            self.assertTrue(resolved.exists())
            self.assertTrue(resolved.is_dir())

    def test_resolve_processed_data_dir_appends_namespace_under_processed_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td) / "save_data"
            processed_root = Path(td) / "processed_data"
            args = argparse.Namespace(path_data=processed_root.as_posix())

            resolved = RL_MAIN._resolve_processed_data_dir(
                args=args,
                data_root=data_root,
                cache_namespace="onnx_frontier_size_64_shared",
            )

            self.assertEqual(resolved, processed_root / "onnx_frontier_size_64_shared")
            self.assertTrue(resolved.exists())
            self.assertTrue(resolved.is_dir())

    def test_prepare_eval_context_rebuilds_only_missing_split_when_cache_is_partial(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_root = root / "save_data"
            model_root = root / "save_model"
            cache_dir = root / "cache" / "processed_data"
            cache_dir.mkdir(parents=True, exist_ok=True)

            query_bundle = {"train": [], "test": []}
            query_bundle_path = root / "query_bundle.json"
            query_bundle_path.write_text(json.dumps(query_bundle), encoding="utf-8")

            cached_train_samples = [{"frontier_label": "common", "sentinel": 11}]
            torch.save(cached_train_samples, cache_dir / "eval_train_samples.pt")

            args = argparse.Namespace(
                build_eval_data=False,
                path_data=cache_dir.as_posix(),
                onnx_frontier_size=64,
                max_size_frontier=64,
                kind_of_data="merged",
                dataset_type="HASHED",
                seed=42,
                max_regular_distance_for_reward=50.0,
                failure_reward_value=-1.0,
                eval_batch_size=128,
                batch_size=128,
                eval_num_workers=0,
            )
            eval_signature = {
                "cache_format": "eval_samples_v1",
                "query_bundle_path": query_bundle_path.as_posix(),
                "kind_of_data": str(args.kind_of_data),
                "dataset_type": str(args.dataset_type),
                "seed": int(args.seed),
                "eval_frontier_size_for_eval_data": int(args.onnx_frontier_size),
                "max_regular_distance_for_reward": float(args.max_regular_distance_for_reward),
                "failure_reward_value": float(args.failure_reward_value),
                "query_counts": {"test": 0, "train": 0},
            }
            (cache_dir / "eval_samples_meta.json").write_text(
                json.dumps(
                    {
                        "signature": eval_signature,
                        "materialization_stats": {
                            "train": {
                                "n_queries": 0,
                                "n_materialized_samples": 1,
                                "n_failed_materialization": 0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                RL_MAIN,
                "_build_or_load_query_bundle",
                return_value=(query_bundle, query_bundle_path, root / "test_data"),
            ), patch.object(
                RL_MAIN,
                "build_frontier_dataloader",
                side_effect=lambda **kwargs: {"n_samples": len(kwargs.get("samples", []))},
            ):
                ctx = RL_MAIN._prepare_eval_context(
                    args=args,
                    data_root=data_root,
                    model_root=model_root,
                )

            self.assertFalse(bool(ctx["eval_samples_loaded_from_cache"]))
            self.assertEqual(ctx["eval_samples"]["train"], cached_train_samples)
            self.assertEqual(ctx["eval_samples"]["test"], [])
            self.assertTrue((cache_dir / "eval_test_samples.pt").exists())


if __name__ == "__main__":
    unittest.main()
