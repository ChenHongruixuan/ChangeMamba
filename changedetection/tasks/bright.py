import torch
import torch.nn.functional as F

import changedetection.utils_func.lovasz_loss as L
from changedetection.datasets import build_train_loader
from changedetection.engine import BaseTrainer
from changedetection.evaluation import BinaryChangeEvaluator, DamageClassificationEvaluator, MultiClassEvaluator
from changedetection.logging_utils import format_log_block
from changedetection.models.ChangeMambaMMBDA import ChangeMambaMMBDA
from changedetection.script.script_utils import get_vssm_kwargs


class BRIGHTTrainer(BaseTrainer):
    task_name = "bright"

    def __init__(self, args):
        self.class_weights = None
        super().__init__(args)
        self.class_weights = torch.tensor([1, 1, 1, 1], dtype=torch.float32, device=self.device)

    def build_model(self, config):
        return ChangeMambaMMBDA(
            output_building=2,
            output_damage=4,
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
        ce_loss_clf = F.cross_entropy(output_clf, labels_clf, weight=self.class_weights, ignore_index=255)
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
        evaluator_clf = DamageClassificationEvaluator(num_classes=4)
        evaluator_total = MultiClassEvaluator(num_classes=4)
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
                evaluator_total.add_batch(labels_clf_np, pred_clf)

        loc_metrics = evaluator_loc.compute()
        damage_metrics = evaluator_clf.compute()
        total_metrics = evaluator_total.compute()
        return {
            "loc_f1": loc_metrics.f1,
            "clf_f1": damage_metrics.harmonic_mean_f1,
            "oa": total_metrics.oa,
            "miou": total_metrics.miou,
            "sub_iou": total_metrics.iou_per_class,
        }

    def selection_metric(self, eval_results):
        return eval_results["Validation"]["miou"]

    def format_eval_result(self, split_name, iteration, total_iterations, metrics):
        return format_log_block(
            f"EVAL {split_name}",
            {
                "loc_F1_pct": 100 * metrics["loc_f1"],
                "clf_F1_pct": 100 * metrics["clf_f1"],
                "OA_pct": 100 * metrics["oa"],
                "mIoU_pct": 100 * metrics["miou"],
                "sub_IoU_pct": 100 * metrics["sub_iou"],
            },
            meta={"iter": f"{iteration}/{total_iterations}"},
        )

    def format_best_result(self, best_record):
        validation = best_record["results"]["Validation"]
        test = best_record["results"]["Test"]
        validation_block = format_log_block(
            "BEST Validation",
            {
                "loc_F1_pct": 100 * validation["loc_f1"],
                "clf_F1_pct": 100 * validation["clf_f1"],
                "OA_pct": 100 * validation["oa"],
                "mIoU_pct": 100 * validation["miou"],
                "sub_IoU_pct": 100 * validation["sub_iou"],
            },
            meta={"iter": best_record["iteration"], "score": best_record["score"]},
        )
        test_block = format_log_block(
            "BEST Test",
            {
                "loc_F1_pct": 100 * test["loc_f1"],
                "clf_F1_pct": 100 * test["clf_f1"],
                "OA_pct": 100 * test["oa"],
                "mIoU_pct": 100 * test["miou"],
                "sub_IoU_pct": 100 * test["sub_iou"],
            },
            meta={"iter": best_record["iteration"], "score": best_record["score"]},
        )
        return f"{validation_block}\n{test_block}"
