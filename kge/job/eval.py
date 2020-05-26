import time
from typing import Any, Dict

import torch
from kge import Config, Dataset

from kge.job import Job, TrainingJob


class EvaluationJob(Job):
    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job)

        self.config = config
        self.dataset = dataset
        self.model = model
        self.batch_size = config.get("eval.batch_size")
        self.device = self.config.get("job.device")
        self.config.check("train.trace_level", ["example", "batch", "epoch"])
        self.trace_examples = self.config.get("eval.trace_level") == "example"
        self.trace_batch = (
            self.trace_examples or self.config.get("train.trace_level") == "batch"
        )
        self.eval_split = self.config.get("eval.split")
        self.filter_splits = self.config.get("eval.filter_splits")
        if self.eval_split not in self.filter_splits:
            self.filter_splits.append(self.eval_split)
        self.epoch = -1

        self.verbose = True
        self.is_prepared = False

        #: Hooks run after training for an epoch.
        #: Signature: job, trace_entry
        self.post_epoch_hooks = []

        #: Hooks run before starting a batch.
        #: Signature: job
        self.pre_batch_hooks = []

        #: Hooks run before outputting the trace of a batch. Can modify trace entry.
        #: Signature: job, trace_entry
        self.post_batch_trace_hooks = []

        #: Hooks run before outputting the trace of an epoch. Can modify trace entry.
        #: Signature: job, trace_entry
        self.post_epoch_trace_hooks = []

        #: Signature: job, trace_entry
        self.post_valid_hooks = []

        #: Hooks after computing the ranks for each batch entry.
        #: Signature: job, trace_entry
        self.hist_hooks = [hist_all]
        if config.get("eval.metrics_per.head_and_tail"):
            self.hist_hooks.append(hist_per_head_and_tail)
        if config.get("eval.metrics_per.relation_type"):
            self.hist_hooks.append(hist_per_relation_type)
        if config.get("eval.metrics_per.argument_frequency"):
            self.hist_hooks.append(hist_per_frequency_percentile)

        # Add the training loss as a default to every evaluation job
        # TODO: create AggregatingEvaluationsJob that runs and aggregates a list
        #  of EvaluationAjobs, such that users can configure combinations of
        #  EvalJobs themselves. Then this can be removed.
        #  See https://github.com/uma-pi1/kge/issues/102
        if not isinstance(self, TrainingLossEvaluationJob):
            self.eval_train_loss_job = TrainingLossEvaluationJob(
                config, dataset, parent_job=self, model=model
            )
            self.eval_train_loss_job.verbose = False
            self.post_epoch_trace_hooks.append(
                lambda job, trace: trace.update(
                    avg_loss=self.eval_train_loss_job.run()["avg_loss"]
                )
            )

        # all done, run job_created_hooks if necessary
        if self.__class__ == EvaluationJob:
            for f in Job.job_created_hooks:
                f(self)

    @staticmethod
    def create(config, dataset, parent_job=None, model=None):
        """Factory method to create an evaluation job """
        from kge.job import EntityRankingJob, EntityPairRankingJob

        # create the job
        if config.get("eval.type") == "entity_ranking":
            return EntityRankingJob(config, dataset, parent_job=parent_job, model=model)
        elif config.get("eval.type") == "entity_pair_ranking":
            return EntityPairRankingJob(
                config, dataset, parent_job=parent_job, model=model
            )
        elif config.get("eval.type") == "training_loss":
            return TrainingLossEvaluationJob(
                config, dataset, parent_job=parent_job, model=model
            )
        else:
            raise ValueError("eval.type")

    def _prepare(self):
        """Prepare this job for running. Guaranteed to be called exactly once
        """
        raise NotImplementedError

    def run(self) -> Dict[str, Any]:

        if not self.is_prepared:
            self._prepare()
            self.model.prepare_job(self)  # let the model add some hooks
            self.is_prepared = True

        was_training = self.model.training
        self.model.eval()
        self.config.log(
            "Evaluating on "
            + self.eval_split
            + " data (epoch {})...".format(self.epoch),
            echo=self.verbose,
        )

        trace_entry = self._run()

        # if validation metric is not present, try to compute it
        metric_name = self.config.get("valid.metric")
        if metric_name not in trace_entry:
            trace_entry[metric_name] = eval(
                self.config.get("valid.metric_expr"),
                None,
                dict(config=self.config, **trace_entry),
            )

        for f in self.post_epoch_trace_hooks:
            f(self, trace_entry)

        # write out trace
        trace_entry = self.trace(
            **trace_entry, echo=self.verbose, echo_prefix="  ", log=True
        )

        # reset model and return metrics
        if was_training:
            self.model.train()
        self.config.log(
            "Finished evaluating on " + self.eval_split + " split.", echo=self.verbose
        )

        for f in self.post_valid_hooks:
            f(self, trace_entry)

        return trace_entry

    def _run(self) -> Dict[str, Any]:
        """ Compute evaluation metrics, output results to trace file """
        raise NotImplementedError

    def resume(self, checkpoint_file=None):
        """Load model state from last or specified checkpoint."""
        # load model
        from kge.job import TrainingJob

        training_job = TrainingJob.create(self.config, self.dataset)
        training_job.resume(checkpoint_file)
        self.model = training_job.model
        self.epoch = training_job.epoch
        self.resumed_from_job_id = training_job.resumed_from_job_id
        self.trace(
            event="job_resumed", epoch=self.epoch, checkpoint_file=checkpoint_file
        )


class TrainingLossEvaluationJob(EvaluationJob):
    """ Entity ranking evaluation protocol """

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        self.is_prepared = True

        train_job_on_eval_split_config = config.clone()
        train_job_on_eval_split_config.set("train.split", self.eval_split)
        train_job_on_eval_split_config.set(
            "negative_sampling.filtering.splits",
            [self.config.get("train.split"), self.eval_split] + ["valid"]
            if self.eval_split == "test"
            else [],
        )
        train_job_on_eval_split_config.set(
            "KvsAll.label_splits",
            [self.config.get("train.split"), self.eval_split] + ["valid"]
            if self.eval_split == "test"
            else [],
        )
        self._train_job = TrainingJob.create(
            config=train_job_on_eval_split_config,
            parent_job=self,
            dataset=dataset,
            initialize_for_forward_only=True,
        )
        self._train_job.model = model
        self._train_job_verbose = False

        if self.__class__ == TrainingLossEvaluationJob:
            for f in Job.job_created_hooks:
                f(self)

    @torch.no_grad()
    def _run(self) -> Dict[str, Any]:

        epoch_time = -time.time()

        self.epoch = self.parent_job.epoch
        epoch_time += time.time()

        train_trace_entry = self._train_job.run_epoch(verbose=self._train_job_verbose)
        # compute trace
        trace_entry = dict(
            type="training_loss",
            scope="epoch",
            split=self.eval_split,
            epoch=self.epoch,
            epoch_time=epoch_time,
            event="eval_completed",
            avg_loss=train_trace_entry["avg_loss"],
        )

        return trace_entry


# HISTOGRAM COMPUTATION ###############################################################


def __initialize_hist(hists, key, job):
    """If there is no histogram with given `key` in `hists`, add an empty one."""
    if key not in hists:
        hists[key] = torch.zeros(
            [job.dataset.num_entities()],
            device=job.config.get("job.device"),
            dtype=torch.float,
        )


def hist_all(hists, s, p, o, s_ranks, o_ranks, job, **kwargs):
    """Create histogram of all subject/object ranks (key: "all").

    `hists` a dictionary of histograms to update; only key "all" will be affected. `s`,
    `p`, `o` are true triples indexes for the batch. `s_ranks` and `o_ranks` are the
    rank of the true answer for (?,p,o) and (s,p,?) obtained from a model.

    """
    __initialize_hist(hists, "all", job)
    hist = hists["all"]
    for r in o_ranks:
        hist[r] += 1
    for r in s_ranks:
        hist[r] += 1


def hist_per_head_and_tail(hists, s, p, o, s_ranks, o_ranks, job, **kwargs):
    __initialize_hist(hists, "head", job)
    hist = hists["head"]
    for r in s_ranks:
        hist[r] += 1

    __initialize_hist(hists, "tail", job)
    hist = hists["tail"]
    for r in o_ranks:
        hist[r] += 1


def hist_per_relation_type(hists, s, p, o, s_ranks, o_ranks, job, **kwargs):
    for rel_type, rels in job.dataset.index("relations_per_type").items():
        __initialize_hist(hists, rel_type, job)
        mask = [_p in rels for _p in p.tolist()]
        for r, m in zip(o_ranks, mask):
            if m:
                hists[rel_type][r] += 1
        for r, m in zip(s_ranks, mask):
            if m:
                hists[rel_type][r] += 1


def hist_per_frequency_percentile(hists, s, p, o, s_ranks, o_ranks, job, **kwargs):
    # initialize
    frequency_percs = job.dataset.index("frequency_percentiles")
    for arg, percs in frequency_percs.items():
        for perc, value in percs.items():
            __initialize_hist(hists, "{}_{}".format(arg, perc), job)

    # go
    for perc in frequency_percs["subject"].keys():  # same for relation and object
        for r, m_s, m_r in zip(
            s_ranks,
            [id in frequency_percs["subject"][perc] for id in s.tolist()],
            [id in frequency_percs["relation"][perc] for id in p.tolist()],
        ):
            if m_s:
                hists["{}_{}".format("subject", perc)][r] += 1
            if m_r:
                hists["{}_{}".format("relation", perc)][r] += 1
        for r, m_o, m_r in zip(
            o_ranks,
            [id in frequency_percs["object"][perc] for id in o.tolist()],
            [id in frequency_percs["relation"][perc] for id in p.tolist()],
        ):
            if m_o:
                hists["{}_{}".format("object", perc)][r] += 1
            if m_r:
                hists["{}_{}".format("relation", perc)][r] += 1
