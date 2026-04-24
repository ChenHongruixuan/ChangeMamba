import os
import sys
import time

import torch
import torch.optim as optim
from tqdm import tqdm

from changedetection.checkpoints import (
    format_checkpoint_load_report,
    load_model_weights,
    resume_training_state,
    save_training_checkpoint,
)
from changedetection.configs.config import get_config
from changedetection.datasets import build_eval_loader
from changedetection.logging_utils import format_log_block, is_interactive_stream
from changedetection.tasks.metadata import get_task_runtime_spec


class BaseTrainer:
    task_name = None
    log_interval = 10
    eval_interval = 500
    best_checkpoint_name = "best_model.pth"
    latest_checkpoint_name = "latest.pth"
    save_iteration_checkpoints = False

    def __init__(self, args):
        self.args = args
        self.config = get_config(args)
        self.device = torch.device("cuda" if getattr(args, "cuda", True) and torch.cuda.is_available() else "cpu")
        self.use_progress_bar = is_interactive_stream(sys.stderr)
        self.runtime_spec = get_task_runtime_spec(self.task_name)
        if self.runtime_spec is not None:
            self.eval_interval = self.runtime_spec.eval_interval
        self.model_checkpoint_path = self._resolve_model_checkpoint_path()
        self.resume_training_path = self._resolve_resume_training_path()
        if self.model_checkpoint_path is not None and self.resume_training_path is not None:
            raise ValueError(
                "Both model checkpoint loading and training resume were requested. "
                "Use --model_checkpoint_path for weight-only initialization or --resume_training_path for full "
                "training resume."
            )
        self.start_iteration = int(getattr(args, "start_iter", 0) or 0)
        self.best_score = None
        self.best_record = None

        self.train_loader = self.build_train_loader()
        self.eval_loaders = self.build_eval_loaders()
        self.model = self.build_model(self.config).to(self.device)
        if self.model_checkpoint_path is not None:
            load_info = load_model_weights(self.model, self.model_checkpoint_path)
            self.emit_log(format_checkpoint_load_report(load_info, title="INIT Model"))
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        self.model_save_path = self._resolve_model_save_path()
        os.makedirs(self.model_save_path, exist_ok=True)

        if self.resume_training_path is not None:
            self._resume_from_checkpoint()

    def build_model(self, config):
        raise NotImplementedError

    def build_train_loader(self):
        raise NotImplementedError

    def build_eval_loaders(self):
        return {}

    def _resolve_model_checkpoint_path(self):
        checkpoint_path = getattr(self.args, "model_checkpoint_path", None)
        if checkpoint_path:
            return checkpoint_path
        config_checkpoint = getattr(self.config.MODEL, "CHECKPOINT", "")
        if config_checkpoint:
            return config_checkpoint
        config_resume = getattr(self.config.MODEL, "RESUME", "")
        return config_resume or None

    def _resolve_resume_training_path(self):
        resume_path = getattr(self.args, "resume_training_path", None)
        if resume_path:
            return resume_path
        config_resume = getattr(self.config.MODEL, "RESUME", "")
        return config_resume or None

    def _resolve_model_save_path(self):
        if self.resume_training_path and os.path.isfile(self.resume_training_path):
            return os.path.dirname(os.path.abspath(self.resume_training_path))
        return os.path.join(
            self.args.model_param_path,
            self.args.dataset,
            f"{self.args.model_type}_{time.time()}",
        )

    def _build_loader_from_spec(self, loader_spec):
        return build_eval_loader(
            self.args.dataset,
            **loader_spec.resolve(self.args),
        )

    def build_runtime_eval_loaders(self):
        if self.runtime_spec is None:
            return {}
        return {
            split_name: self._build_loader_from_spec(loader_spec)
            for split_name, loader_spec in self.runtime_spec.eval_loaders.items()
        }

    def _resume_from_checkpoint(self):
        resume_state = resume_training_state(
            self.resume_training_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.start_iteration = max(self.start_iteration, resume_state["iteration"])
        self.best_score = resume_state["best_score"]
        self.best_record = resume_state["best_record"]

        load_info = resume_state["load_info"]
        self.emit_log(format_checkpoint_load_report(load_info, title="RESUME Model"))
        self.emit_log(
            format_log_block(
                "RESUME State",
                {
                    "iteration": self.start_iteration,
                    "best_score": resume_state["best_score"] if resume_state["best_score"] is not None else "None",
                    "optimizer_loaded": resume_state["optimizer_loaded"],
                    "scheduler_loaded": resume_state["scheduler_loaded"],
                },
            )
        )
        if resume_state["optimizer_error"] is not None:
            self.emit_log(resume_state["optimizer_error"])
        if resume_state["scheduler_error"] is not None:
            self.emit_log(resume_state["scheduler_error"])
        if not any(
            (
                resume_state["has_optimizer_state"],
                resume_state["has_scheduler_state"],
                resume_state["has_iteration_state"],
            )
        ):
            self.emit_log(
                "Resume checkpoint contained model weights only. Optimizer/scheduler/iteration state were not "
                "found, so training restarts from iteration 0. Use --model_checkpoint_path for weight-only "
                "initialization."
            )

    def build_optimizer(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def build_scheduler(self):
        return None

    def train_step(self, batch):
        raise NotImplementedError

    def evaluate_loader(self, split_name, data_loader):
        raise NotImplementedError

    def selection_metric(self, eval_results):
        raise NotImplementedError

    def format_train_log(self, iteration, total_iterations, log_items):
        return format_log_block(
            "TRAIN",
            log_items,
            meta={"iter": f"{iteration}/{total_iterations}"},
        )

    def format_eval_result(self, split_name, iteration, total_iterations, metrics):
        return format_log_block(
            f"EVAL {split_name}",
            metrics,
            meta={"iter": f"{iteration}/{total_iterations}"},
        )

    def format_best_result(self, best_record):
        return str(best_record)

    def emit_log(self, message):
        if self.use_progress_bar:
            tqdm.write(message)
        else:
            print(message, flush=True)

    def after_optimizer_step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def save_checkpoint(self, filename, iteration, best_score, best_record, extra_state=None):
        checkpoint_path = os.path.join(self.model_save_path, filename)
        save_training_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            iteration=iteration,
            best_score=best_score,
            best_record=best_record,
            task_name=self.task_name,
            config=self.config,
            args=self.args,
            extra_state=extra_state,
        )
        return checkpoint_path

    def training(self):
        best_record = self.best_record
        best_score = self.best_score

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        total_iterations = len(self.train_loader)
        self.model.train()
        iterator = self.train_loader
        if self.use_progress_bar:
            iterator = tqdm(self.train_loader, total=total_iterations, dynamic_ncols=True, leave=False)

        for iteration, batch in enumerate(iterator, start=1):
            if iteration <= self.start_iteration:
                continue

            step_output = self.train_step(batch)
            if step_output is None:
                continue

            loss = step_output["loss"]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.after_optimizer_step()

            if iteration % self.log_interval == 0:
                self.emit_log(self.format_train_log(iteration, total_iterations, step_output["log_items"]))

            if self.eval_loaders and iteration % self.eval_interval == 0:
                self.model.eval()
                eval_results = {}
                for split_name, loader in self.eval_loaders.items():
                    eval_results[split_name] = self.evaluate_loader(split_name, loader)
                    self.emit_log(
                        self.format_eval_result(
                            split_name,
                            iteration,
                            total_iterations,
                            eval_results[split_name],
                        )
                    )

                score = self.selection_metric(eval_results)
                is_best = best_score is None or score > best_score
                if is_best:
                    best_checkpoint_path = self.save_checkpoint(
                        self.best_checkpoint_name,
                        iteration,
                        score,
                        {
                            "iteration": iteration,
                            "score": score,
                            "results": eval_results,
                            "checkpoint_path": os.path.join(self.model_save_path, self.best_checkpoint_name),
                        },
                        extra_state={"last_eval_results": eval_results},
                    )
                    best_score = score
                    best_record = {
                        "iteration": iteration,
                        "score": score,
                        "results": eval_results,
                        "checkpoint_path": best_checkpoint_path,
                    }
                self.save_checkpoint(
                    self.latest_checkpoint_name,
                    iteration,
                    best_score,
                    best_record,
                    extra_state={"last_eval_results": eval_results},
                )
                if self.save_iteration_checkpoints:
                    self.save_checkpoint(
                        f"iter_{iteration}.pth",
                        iteration,
                        best_score,
                        best_record,
                        extra_state={"last_eval_results": eval_results},
                    )
                self.model.train()

        if best_record is not None:
            self.emit_log(self.format_best_result(best_record))
        else:
            self.emit_log("Training finished without producing an evaluated best checkpoint.")


class BaseInferer:
    task_name = None
    show_progress = False

    def __init__(self, args):
        self.args = args
        self.config = get_config(args)
        self.device = torch.device("cuda" if getattr(args, "cuda", True) and torch.cuda.is_available() else "cpu")
        self.use_progress_bar = is_interactive_stream(sys.stderr)
        self.runtime_spec = get_task_runtime_spec(self.task_name)
        self.model_checkpoint_path = self._resolve_model_checkpoint_path()

        self.model = self.build_model(self.config).to(self.device)
        if self.model_checkpoint_path is not None:
            load_info = load_model_weights(self.model, self.model_checkpoint_path)
            self.emit_log(format_checkpoint_load_report(load_info, title="CHECKPOINT Load"))
        self.model.eval()

        self.data_loader = self.build_data_loader()
        self.prepare_output_dirs()

    def build_model(self, config):
        raise NotImplementedError

    def build_data_loader(self):
        raise NotImplementedError

    def _resolve_model_checkpoint_path(self):
        checkpoint_path = getattr(self.args, "model_checkpoint_path", None)
        if checkpoint_path:
            return checkpoint_path
        config_checkpoint = getattr(self.config.MODEL, "CHECKPOINT", "")
        return config_checkpoint or None

    def build_runtime_data_loader(self):
        if self.runtime_spec is None or self.runtime_spec.infer_loader is None:
            raise NotImplementedError("No infer loader spec registered for this task.")
        return build_eval_loader(
            self.args.dataset,
            **self.runtime_spec.infer_loader.resolve(self.args),
        )

    def prepare_output_dirs(self):
        return None

    def emit_log(self, message):
        if self.use_progress_bar:
            tqdm.write(message)
        else:
            print(message, flush=True)

    def infer_batch(self, batch):
        raise NotImplementedError

    def finish(self):
        return None

    def infer(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        iterator = self.data_loader
        if self.show_progress and self.use_progress_bar:
            iterator = tqdm(self.data_loader, dynamic_ncols=True, leave=False)
        with torch.no_grad():
            for batch in iterator:
                self.infer_batch(batch)
        self.finish()
