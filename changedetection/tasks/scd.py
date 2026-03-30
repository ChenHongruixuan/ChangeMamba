import os

import imageio
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import changedetection.utils_func.lovasz_loss as L
from changedetection.datasets import build_train_loader
from changedetection.engine import BaseInferer, BaseTrainer
from changedetection.evaluation import SemanticChangeEvaluator
from changedetection.logging_utils import format_log_block
from changedetection.models.ChangeMambaSCD import ChangeMambaSCD
from changedetection.script.script_utils import get_vssm_kwargs, map_labels_to_colors
from changedetection.tasks.metadata import SECOND_COLOR_MAP, SECOND_LABELS


class SCDTrainer(BaseTrainer):
    task_name = "scd"

    def build_model(self, config):
        return ChangeMambaSCD(
            output_cd=2,
            output_clf=7,
            pretrained=self.args.pretrained_weight_path,
            **get_vssm_kwargs(config),
        )

    def build_train_loader(self):
        return build_train_loader(self.args)

    def build_eval_loaders(self):
        return self.build_runtime_eval_loaders()

    def build_scheduler(self):
        return StepLR(self.optimizer, step_size=10000, gamma=0.5)

    def train_step(self, batch):
        pre_change_imgs, post_change_imgs, label_cd, label_clf_t1, label_clf_t2, _ = batch
        pre_change_imgs = pre_change_imgs.to(self.device)
        post_change_imgs = post_change_imgs.to(self.device)
        label_cd = label_cd.to(self.device).long()
        label_clf_t1 = label_clf_t1.to(self.device).long()
        label_clf_t2 = label_clf_t2.to(self.device).long()

        label_clf_t1 = label_clf_t1.clone()
        label_clf_t2 = label_clf_t2.clone()
        label_clf_t1[label_clf_t1 == 0] = 255
        label_clf_t2[label_clf_t2 == 0] = 255

        output_cd, output_semantic_t1, output_semantic_t2 = self.model(pre_change_imgs, post_change_imgs)
        ce_loss_cd = F.cross_entropy(output_cd, label_cd, ignore_index=255)
        lovasz_loss_cd = L.lovasz_softmax(F.softmax(output_cd, dim=1), label_cd, ignore=255)
        ce_loss_clf_t1 = F.cross_entropy(output_semantic_t1, label_clf_t1, ignore_index=255)
        lovasz_loss_clf_t1 = L.lovasz_softmax(F.softmax(output_semantic_t1, dim=1), label_clf_t1, ignore=255)
        ce_loss_clf_t2 = F.cross_entropy(output_semantic_t2, label_clf_t2, ignore_index=255)
        lovasz_loss_clf_t2 = L.lovasz_softmax(F.softmax(output_semantic_t2, dim=1), label_clf_t2, ignore=255)

        similarity_mask = (label_clf_t1 == 255).float().unsqueeze(1).expand_as(output_semantic_t1)
        similarity_loss = F.mse_loss(
            F.softmax(output_semantic_t1, dim=1) * similarity_mask,
            F.softmax(output_semantic_t2, dim=1) * similarity_mask,
            reduction="mean",
        )

        final_loss = (
            ce_loss_cd
            + 0.5 * (ce_loss_clf_t1 + ce_loss_clf_t2 + 0.5 * similarity_loss)
            + 0.75 * (lovasz_loss_cd + 0.5 * (lovasz_loss_clf_t1 + lovasz_loss_clf_t2))
        )

        clf_loss = (
            ce_loss_clf_t1 + ce_loss_clf_t2 + lovasz_loss_clf_t1 + lovasz_loss_clf_t2
        ) / 2
        return {
            "loss": final_loss,
            "log_items": {
                "cd_loss": (ce_loss_cd + lovasz_loss_cd).item(),
                "clf_loss": clf_loss.item(),
            },
        }

    def evaluate_loader(self, split_name, data_loader):
        evaluator = SemanticChangeEvaluator(num_class=37)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            for pre_change_imgs, post_change_imgs, labels_cd, labels_clf_t1, labels_clf_t2, _ in data_loader:
                pre_change_imgs = pre_change_imgs.to(self.device)
                post_change_imgs = post_change_imgs.to(self.device)
                labels_cd = labels_cd.to(self.device).long()
                labels_clf_t1 = labels_clf_t1.to(self.device).long()
                labels_clf_t2 = labels_clf_t2.to(self.device).long()

                output_cd, output_semantic_t1, output_semantic_t2 = self.model(pre_change_imgs, post_change_imgs)
                labels_cd_np = labels_cd.cpu().numpy()
                labels_a = labels_clf_t1.cpu().numpy()
                labels_b = labels_clf_t2.cpu().numpy()

                change_mask = torch.argmax(output_cd, dim=1).cpu().numpy()
                preds_a = torch.argmax(output_semantic_t1, dim=1).cpu().numpy()
                preds_b = torch.argmax(output_semantic_t2, dim=1).cpu().numpy()

                preds_scd = (preds_a - 1) * 6 + preds_b
                preds_scd[change_mask == 0] = 0

                labels_scd = (labels_a - 1) * 6 + labels_b
                labels_scd[labels_cd_np == 0] = 0

                for pred_scd, label_scd in zip(preds_scd, labels_scd):
                    evaluator.add_batch(pred_scd, label_scd)

        metrics = evaluator.compute()
        return {
            "kappa": metrics.kappa,
            "fscd": metrics.fscd,
            "miou": metrics.miou,
            "sek": metrics.sek,
            "oa": metrics.oa,
        }

    def selection_metric(self, eval_results):
        return eval_results["Validation"]["sek"]

    def format_eval_result(self, split_name, iteration, total_iterations, metrics):
        return format_log_block(
            f"EVAL {split_name}",
            {
                "Kappa": metrics["kappa"],
                "Fscd": metrics["fscd"],
                "OA": metrics["oa"],
                "mIoU": metrics["miou"],
                "SeK": metrics["sek"],
            },
            meta={"iter": f"{iteration}/{total_iterations}"},
        )

    def format_best_result(self, best_record):
        metrics = best_record["results"]["Validation"]
        return format_log_block(
            "BEST Validation",
            {
                "Kappa": metrics["kappa"],
                "Fscd": metrics["fscd"],
                "OA": metrics["oa"],
                "mIoU": metrics["miou"],
                "SeK": metrics["sek"],
            },
            meta={"iter": best_record["iteration"], "score": best_record["score"]},
        )


class SCDInferer(BaseInferer):
    task_name = "scd"

    def build_model(self, config):
        return ChangeMambaSCD(
            output_cd=2,
            output_clf=7,
            pretrained=self.args.pretrained_weight_path,
            **get_vssm_kwargs(config),
        )

    def build_data_loader(self):
        return self.build_runtime_data_loader()

    def prepare_output_dirs(self):
        self.change_map_T1_saved_path = os.path.join(
            self.args.result_saved_path,
            self.args.dataset,
            self.args.model_type,
            "change_map_T1",
        )
        self.change_map_T2_saved_path = os.path.join(
            self.args.result_saved_path,
            self.args.dataset,
            self.args.model_type,
            "change_map_T2",
        )
        os.makedirs(self.change_map_T1_saved_path, exist_ok=True)
        os.makedirs(self.change_map_T2_saved_path, exist_ok=True)

    def infer_batch(self, batch):
        pre_change_imgs, post_change_imgs, _, _, _, names = batch
        pre_change_imgs = pre_change_imgs.to(self.device)
        post_change_imgs = post_change_imgs.to(self.device)

        output_cd, output_semantic_t1, output_semantic_t2 = self.model(pre_change_imgs, post_change_imgs)
        change_mask = torch.argmax(output_cd, dim=1)
        preds_a = (torch.argmax(output_semantic_t1, dim=1) * change_mask.squeeze().long()).cpu().numpy()
        preds_b = (torch.argmax(output_semantic_t2, dim=1) * change_mask.squeeze().long()).cpu().numpy()

        image_name = os.path.splitext(names[0])[0] + ".png"
        change_map_t1 = map_labels_to_colors(preds_a.squeeze(), SECOND_COLOR_MAP, SECOND_LABELS)
        change_map_t2 = map_labels_to_colors(preds_b.squeeze(), SECOND_COLOR_MAP, SECOND_LABELS)
        imageio.imwrite(os.path.join(self.change_map_T1_saved_path, image_name), change_map_t1)
        imageio.imwrite(os.path.join(self.change_map_T2_saved_path, image_name), change_map_t2)

    def finish(self):
        self.emit_log("Inference stage is done!")
