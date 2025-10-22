

import os
import math
import numpy as np
import torch
import scipy.signal as signal
import pytorch_lightning as pl
import metrics
import time

class CSMFCC_FN:
    def __call__(self, ref, deg):
        return metrics.mfcc_cosine_similarity(ref, deg, fs=8000)
    @staticmethod
    def min():
        return -1.0

class NBPESQ_FN:
    def __call__(self, ref, deg):
        return metrics.nbpesq(ref, deg, fs=8000)
    @staticmethod
    def min():
        return 1.0 
class DNSMOS_OVRL:
    def __call__(self, ref, deg):
        return metrics.DNSMOS_OVRL(deg, fs=8000)
        
    @staticmethod
    def min():
        return 1.0 

class ESTOI_FN:
    def __call__(self, ref, deg):
        return metrics.estoi(ref, deg, fs=8000)

    @staticmethod
    def min():
        return 0.0
class BaseModel(pl.LightningModule):

    def __init__(self):

        super().__init__()
        self.val_outputs = []
        self.val_targets = []
        self.val_tasks = []

        self.test_outputs = []
        self.test_targets = []
        self.test_tasks = []
        
        self.heavy_eval = False

        self.csmfcc_fn = CSMFCC_FN()
        self.pesq_fn = NBPESQ_FN()
        self.dnsmos_fn = DNSMOS_OVRL()
        self.estoi_fn = ESTOI_FN()

    def common_step(self, batch, batch_idx, mode="train"):
        
        clean = batch["clean"]
        noisy = batch["recorded"]
        task = batch["task"]
        enhanced = self.forward(noisy)

        return enhanced, clean, task

    def training_step(self, batch, batch_idx):
        enhanced, clean, task = self.common_step(batch, batch_idx, mode="train")
        loss = self.loss_function(clean, enhanced)

        self.log(f'train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):

        enhanced, clean, task = self.common_step(batch, batch_idx, mode="val")
        loss = self.loss_function(clean, enhanced)
        self.log(f'val/loss', loss)
        if self.heavy_eval:
            self.val_outputs.append(enhanced.detach().cpu().numpy())
            self.val_targets.append(clean.detach().cpu().numpy())
            self.val_tasks.append(task)

        return loss

    def test_step(self, batch, batch_idx):
        enhanced, clean, task = self.common_step(batch, batch_idx, mode="test")
        self.test_outputs.append(enhanced.detach().cpu().numpy())
        self.test_targets.append(clean.detach().cpu().numpy())
        self.test_tasks.append(task)

    def metrics_evaluation(self, mode, outputs, targets, tasks):

        # Compute metrics (loop over utterances if necessary)
        pesq_vals, estoi_vals, dnsmos_vals, csmfcc_vals= [], [], [], []

        task1_scores_vals = []
        task2_scores_vals = []

        def evaluate_metrics_per_batch(fn, measure_time=False):

            if measure_time:
                torch.cuda.synchronize()  # make sure all prior ops are done
                start = time.perf_counter()
            output = []
            for batch_ref, batch_deg in zip(targets, outputs):
                for ref, deg in zip(batch_ref, batch_deg):
                    try:
                        val = fn(ref, deg)
                    except Exception as e: 
                        # print(e)
                        val = fn.min()
                    output.append(val)

            if measure_time:
                torch.cuda.synchronize()  # wait for ops to finish
                end = time.perf_counter()
                print(f"Elapsed time: {end - start:.6f} seconds")

            return output
        


        csmfcc_vals = evaluate_metrics_per_batch(self.csmfcc_fn, measure_time=True)

        
        pesq_vals = evaluate_metrics_per_batch(self.pesq_fn, measure_time=True)

        dnsmos_vals = evaluate_metrics_per_batch(self.dnsmos_fn, measure_time=True)
        
        estoi_vals = evaluate_metrics_per_batch(self.estoi_fn, measure_time=True)


        task_vals = []
        for batch_task in tasks:
            for task in batch_task:
                task_vals.append(task)

        assert len(task_vals) == len(csmfcc_vals), f"{len(task_vals)} vs {len(csmfcc_vals)}"

        for pesq_val, dnsmos_val, estoi_val, csmfcc_val, task in \
            zip(pesq_vals, dnsmos_vals, estoi_vals, csmfcc_vals, task_vals):

            dnsmos_n = (dnsmos_val - 1.0) / (5.0 - 1.0)
            pesq_n = (pesq_val - 1.0) / (4.5 - 1.0)

            weighted_score = (dnsmos_n + pesq_n + csmfcc_val + estoi_val ) / 4
            if task == "Task1":
                task1_scores_vals.append(weighted_score)
            elif task == "Task2":
                task2_scores_vals.append(weighted_score)

        # weighted_score_vals * task_weightage 
        # Log averaged results
        self.log(f"{mode}/pesq", torch.tensor(pesq_vals).mean(),
                 prog_bar=True, sync_dist=True)
        self.log(f"{mode}/estoi", torch.tensor(estoi_vals).mean(),
                 prog_bar=True, sync_dist=True)
        self.log(f"{mode}/dnsmos", torch.tensor(dnsmos_vals).mean(),
                 prog_bar=True, sync_dist=True)
        self.log(f"{mode}/csmfcc", torch.tensor(csmfcc_vals).mean(),
                 prog_bar=True, sync_dist=True)
        
        task1_score = torch.tensor(task1_scores_vals).mean()
        task2_score = torch.tensor(task2_scores_vals).mean()
        
        self.log(f"{mode}/task1_score", task1_score,
                prog_bar=True, sync_dist=True)
        self.log(f"{mode}/task2_score", task2_score,
                prog_bar=True, sync_dist=True)
        self.log(f"{mode}/weighted_score", 0.4 * task1_score + 0.6 * task2_score ,
                prog_bar=True, sync_dist=True)



    
    def on_validation_epoch_end(self):

        if self.heavy_eval:
            self.metrics_evaluation("val", 
                self.val_outputs, 
                self.val_targets, 
                self.val_tasks)

            self.val_outputs.clear()
            self.val_targets.clear()
            self.val_tasks.clear()


    def on_test_epoch_end(self):

        self.metrics_evaluation("test", 
            self.test_outputs, 
            self.test_targets, 
            self.test_tasks)

        self.test_outputs.clear()
        self.test_targets.clear()
        self.test_tasks.clear()
       

