import os

import imageio
import torch
import torch.nn.functional as F

import changedetection.utils_func.lovasz_loss as L
from changedetection.datasets import build_train_loader
from changedetection.engine import BaseInferer, BaseTrainer
from changedetection.evaluation import BinaryChangeEvaluator
from changedetection.logging_utils import format_log_block
from changedetection.models.ChangeMambaBCD import ChangeMambaBCD
from changedetection.script.script_utils import get_vssm_kwargs


class BCDTrainer(BaseTrainer):
    task_name = "bcd"

    def build_model(self, config):
        return ChangeMambaBCD(
            pretrained=self.args.encoder_pretrained_path,
            **get_vssm_kwargs(config),
        )

    def build_train_loader(self):
        return build_train_loader(self.args)

    def build_eval_loaders(self):
        return self.build_runtime_eval_loaders()

    def train_step(self, batch):
        pre_change_imgs, post_change_imgs, labels, _ = batch
        pre_change_imgs = pre_change_imgs.to(self.device).float()
        post_change_imgs = post_change_imgs.to(self.device)
        labels = labels.to(self.device).long()

        output = self.model(pre_change_imgs, post_change_imgs)
        ce_loss = F.cross_entropy(output, labels, ignore_index=255)
        lovasz_loss = L.lovasz_softmax(F.softmax(output, dim=1), labels, ignore=255)
        final_loss = ce_loss + 0.75 * lovasz_loss
        return {
            "loss": final_loss,
            "log_items": {"loss": final_loss.item()},
        }

    def evaluate_loader(self, split_name, data_loader):
        evaluator = BinaryChangeEvaluator()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            for pre_change_imgs, post_change_imgs, labels, _ in data_loader:
                pre_change_imgs = pre_change_imgs.to(self.device).float()
                post_change_imgs = post_change_imgs.to(self.device)
                labels = labels.to(self.device).long()

                output = self.model(pre_change_imgs, post_change_imgs)
                predictions = torch.argmax(output, dim=1).cpu().numpy()
                evaluator.add_batch(labels.cpu().numpy(), predictions)

        metrics = evaluator.compute()
        return {
            "recall": metrics.recall,
            "precision": metrics.precision,
            "oa": metrics.oa,
            "f1": metrics.f1,
            "iou": metrics.iou,
            "kappa": metrics.kappa,
        }

    def selection_metric(self, eval_results):
        return eval_results["Validation"]["kappa"]

    def format_eval_result(self, split_name, iteration, total_iterations, metrics):
        return format_log_block(
            f"EVAL {split_name}",
            {
                "Recall": metrics["recall"],
                "Precision": metrics["precision"],
                "OA": metrics["oa"],
                "F1": metrics["f1"],
                "IoU": metrics["iou"],
                "Kappa": metrics["kappa"],
            },
            meta={"iter": f"{iteration}/{total_iterations}"},
        )

    def format_best_result(self, best_record):
        metrics = best_record["results"]["Validation"]
        return format_log_block(
            "BEST Validation",
            {
                "Recall": metrics["recall"],
                "Precision": metrics["precision"],
                "OA": metrics["oa"],
                "F1": metrics["f1"],
                "IoU": metrics["iou"],
                "Kappa": metrics["kappa"],
            },
            meta={"iter": best_record["iteration"], "score": best_record["score"]},
        )


class BCDInferer(BaseInferer):
    task_name = "bcd"

    def __init__(self, args):
        self.evaluator = BinaryChangeEvaluator()
        super().__init__(args)

    def build_model(self, config):
        return ChangeMambaBCD(
            pretrained=self.args.encoder_pretrained_path,
            **get_vssm_kwargs(config),
        )

    def build_data_loader(self):
        return self.build_runtime_data_loader()

    def prepare_output_dirs(self):
        self.change_map_saved_path = os.path.join(
            self.args.result_saved_path,
            self.args.dataset,
            self.args.model_type,
            "change_map",
        )
        os.makedirs(self.change_map_saved_path, exist_ok=True)

    def infer_batch(self, batch):
        pre_change_imgs, post_change_imgs, labels, names = batch
        pre_change_imgs = pre_change_imgs.to(self.device).float()
        post_change_imgs = post_change_imgs.to(self.device)
        labels = labels.to(self.device).long()

        output = self.model(pre_change_imgs, post_change_imgs)
        predictions = torch.argmax(output, dim=1).cpu().numpy()
        self.evaluator.add_batch(labels.cpu().numpy(), predictions)

        image_name = os.path.splitext(names[0])[0] + ".png"
        binary_change_map = predictions.squeeze().astype("uint8")
        binary_change_map[binary_change_map == 1] = 255
        imageio.imwrite(os.path.join(self.change_map_saved_path, image_name), binary_change_map)

    def finish(self):
        metrics = self.evaluator.compute()
        self.emit_log(
            format_log_block(
                "INFER Summary",
                {
                    "Recall": metrics.recall,
                    "Precision": metrics.precision,
                    "OA": metrics.oa,
                    "F1": metrics.f1,
                    "IoU": metrics.iou,
                    "Kappa": metrics.kappa,
                },
            )
        )
        self.emit_log("Inference stage is done!")
