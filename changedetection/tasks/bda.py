import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F

import changedetection.utils_func.lovasz_loss as L
from changedetection.datasets import build_train_loader
from changedetection.engine import BaseInferer, BaseTrainer
from changedetection.evaluation import BinaryChangeEvaluator, DamageClassificationEvaluator
from changedetection.logging_utils import format_log_block
from changedetection.models.ChangeMambaBDA import ChangeMambaBDA
from changedetection.script.script_utils import get_vssm_kwargs, map_labels_to_colors
from changedetection.tasks.metadata import XBD_DAMAGE_COLOR_MAP, XBD_DAMAGE_LABELS


class BDATrainer(BaseTrainer):
    task_name = "bda"

    def build_model(self, config):
        return ChangeMambaBDA(
            output_building=2,
            output_damage=5,
            pretrained=self.args.pretrained_weight_path,
            **get_vssm_kwargs(config),
        )

    def build_train_loader(self):
        return build_train_loader(self.args)

    def build_eval_loaders(self):
        return self.build_runtime_eval_loaders()

    def train_step(self, batch):
        pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = batch
        pre_change_imgs = pre_change_imgs.to(self.device)
        post_change_imgs = post_change_imgs.to(self.device)
        labels_loc = labels_loc.to(self.device).long()
        labels_clf = labels_clf.to(self.device).long()

        if not (labels_clf != 255).any():
            return None

        output_loc, output_clf = self.model(pre_change_imgs, post_change_imgs)
        ce_loss_loc = F.cross_entropy(output_loc, labels_loc, ignore_index=255)
        lovasz_loss_loc = L.lovasz_softmax(F.softmax(output_loc, dim=1), labels_loc, ignore=255)
        ce_loss_clf = F.cross_entropy(output_clf, labels_clf, ignore_index=255)
        lovasz_loss_clf = L.lovasz_softmax(F.softmax(output_clf, dim=1), labels_clf, ignore=255)
        final_loss = ce_loss_loc + ce_loss_clf + 0.5 * lovasz_loss_loc + 0.75 * lovasz_loss_clf

        return {
            "loss": final_loss,
            "log_items": {
                "loc_loss": (ce_loss_loc + lovasz_loss_loc).item(),
                "clf_loss": (ce_loss_clf + lovasz_loss_clf).item(),
            },
        }

    def evaluate_loader(self, split_name, data_loader):
        evaluator_loc = BinaryChangeEvaluator()
        evaluator_clf = DamageClassificationEvaluator(num_classes=5)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            for pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ in data_loader:
                pre_change_imgs = pre_change_imgs.to(self.device)
                post_change_imgs = post_change_imgs.to(self.device)
                labels_loc = labels_loc.to(self.device).long()
                labels_clf = labels_clf.to(self.device).long()

                output_loc, output_clf = self.model(pre_change_imgs, post_change_imgs)
                pred_loc = torch.argmax(output_loc, dim=1).cpu().numpy()
                pred_clf = torch.argmax(output_clf, dim=1).cpu().numpy()
                labels_loc_np = labels_loc.cpu().numpy()
                labels_clf_np = labels_clf.cpu().numpy()

                evaluator_loc.add_batch(labels_loc_np, pred_loc)
                evaluator_clf.add_batch(labels_clf_np[labels_loc_np > 0], pred_clf[labels_loc_np > 0])

        loc_metrics = evaluator_loc.compute()
        damage_metrics = evaluator_clf.compute()
        oa_f1 = 0.3 * loc_metrics.f1 + 0.7 * damage_metrics.harmonic_mean_f1
        return {
            "loc_f1": loc_metrics.f1,
            "clf_f1": damage_metrics.harmonic_mean_f1,
            "oa_f1": oa_f1,
            "sub_f1": damage_metrics.per_class_f1,
        }

    def selection_metric(self, eval_results):
        return eval_results["Validation"]["oa_f1"]

    def format_eval_result(self, split_name, iteration, total_iterations, metrics):
        return format_log_block(
            f"EVAL {split_name}",
            {
                "loc_F1": metrics["loc_f1"],
                "clf_F1": metrics["clf_f1"],
                "oa_F1": metrics["oa_f1"],
                "sub_F1": metrics["sub_f1"],
            },
            meta={"iter": f"{iteration}/{total_iterations}"},
        )

    def format_best_result(self, best_record):
        metrics = best_record["results"]["Validation"]
        return format_log_block(
            "BEST Validation",
            {
                "loc_F1": metrics["loc_f1"],
                "clf_F1": metrics["clf_f1"],
                "oa_F1": metrics["oa_f1"],
                "sub_F1": metrics["sub_f1"],
            },
            meta={"iter": best_record["iteration"], "score": best_record["score"]},
        )


class BDAInferer(BaseInferer):
    task_name = "bda"
    show_progress = True

    def __init__(self, args):
        self.evaluator_loc = BinaryChangeEvaluator()
        self.evaluator_clf = DamageClassificationEvaluator(num_classes=5)
        super().__init__(args)

    def build_model(self, config):
        return ChangeMambaBDA(
            output_building=2,
            output_damage=5,
            pretrained=self.args.pretrained_weight_path,
            **get_vssm_kwargs(config),
        )

    def build_data_loader(self):
        return self.build_runtime_data_loader()

    def prepare_output_dirs(self):
        self.building_map_saved_path = os.path.join(
            self.args.result_saved_path,
            self.args.dataset,
            self.args.model_type,
            "building_localization_map",
        )
        self.damage_map_saved_path = os.path.join(
            self.args.result_saved_path,
            self.args.dataset,
            self.args.model_type,
            "damage_classification_map",
        )
        os.makedirs(self.building_map_saved_path, exist_ok=True)
        os.makedirs(self.damage_map_saved_path, exist_ok=True)

    def infer_batch(self, batch):
        pre_change_imgs, post_change_imgs, labels_loc, labels_clf, names = batch
        pre_change_imgs = pre_change_imgs.to(self.device)
        post_change_imgs = post_change_imgs.to(self.device)
        labels_loc = labels_loc.to(self.device).long()
        labels_clf = labels_clf.to(self.device).long()

        output_loc, output_clf = self.model(pre_change_imgs, post_change_imgs)
        pred_loc = torch.argmax(output_loc, dim=1).cpu().numpy()
        pred_clf = torch.argmax(output_clf, dim=1).cpu().numpy()
        labels_loc_np = labels_loc.cpu().numpy()
        labels_clf_np = labels_clf.cpu().numpy()

        self.evaluator_loc.add_batch(labels_loc_np, pred_loc)
        self.evaluator_clf.add_batch(labels_clf_np[labels_loc_np > 0], pred_clf[labels_loc_np > 0])

        image_name = names[0] + ".png"
        loc_map = pred_loc.squeeze().astype(np.uint8)
        loc_map[loc_map > 0] = 255
        clf_map = map_labels_to_colors(pred_clf.squeeze(), XBD_DAMAGE_COLOR_MAP, XBD_DAMAGE_LABELS)
        clf_map[loc_map == 0] = 0

        imageio.imwrite(os.path.join(self.building_map_saved_path, image_name), loc_map)
        imageio.imwrite(os.path.join(self.damage_map_saved_path, image_name), clf_map)

    def finish(self):
        loc_metrics = self.evaluator_loc.compute()
        damage_metrics = self.evaluator_clf.compute()
        oa_f1 = 0.3 * loc_metrics.f1 + 0.7 * damage_metrics.harmonic_mean_f1
        self.emit_log(
            format_log_block(
                "INFER Summary",
                {
                    "loc_F1": loc_metrics.f1,
                    "clf_F1": damage_metrics.harmonic_mean_f1,
                    "oa_F1": oa_f1,
                    "sub_F1": damage_metrics.per_class_f1,
                },
            )
        )
        self.emit_log("Inference stage is done!")
