"""
Scheduled retraining with drift detection and automatic rollback.

Flow (run from Task Scheduler daily/weekly, or via pipeline CLI):
  1. is_due() checks the schedule against the registry's last save.
  2. retrain() fits a fresh model on current data, walk-forward validates,
     and only promotes it if it beats the active model out-of-sample.
  3. check_drift() watches live performance and feature importances;
     on drift it can roll back to the previous model version.
"""
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd
from loguru import logger

from quant import config
from quant.backtest.validation import evaluate_walk_forward
from quant.journal import JournalLogger
from quant.models.registry import ModelRegistry


class RetrainScheduler:
    def __init__(self, model_name: str = "prob_up",
                 registry: Optional[ModelRegistry] = None,
                 journal: Optional[JournalLogger] = None):
        self.model_name = model_name
        self.registry = registry or ModelRegistry()
        self.journal = journal or JournalLogger()

    # -------------------------------------------------------- schedule
    def is_due(self) -> bool:
        try:
            _, meta = self.registry.load(self.model_name)
        except FileNotFoundError:
            return True
        saved_at = pd.Timestamp(meta.get("saved_at"))
        age = datetime.now(timezone.utc) - saved_at.to_pydatetime()
        limit = timedelta(days=1 if config.RETRAIN_SCHEDULE == "daily" else 7)
        return age >= limit

    # --------------------------------------------------------- retrain
    def retrain(self, model_factory: Callable[[], object],
                X: pd.DataFrame, y: pd.Series,
                n_folds: int = 5) -> Optional[int]:
        """
        Fit + validate a challenger; promote only if it beats the champion.

        Returns the promoted version number, or None if the challenger lost.
        """
        folds = evaluate_walk_forward(model_factory, X, y, n_folds=n_folds)
        if folds.empty:
            logger.warning("walk-forward produced no valid folds; not retraining")
            return None
        challenger_auc = float(folds["auc"].mean())

        champion_auc = None
        try:
            _, champ_meta = self.registry.load(self.model_name)
            champion_auc = champ_meta.get("walk_forward_auc")
        except FileNotFoundError:
            pass

        model = model_factory()
        model.fit(X, y)
        importances = {}
        if hasattr(model, "feature_importances"):
            importances = model.feature_importances().head(20).to_dict()

        version = self.registry.save(self.model_name, model, {
            "walk_forward_auc": challenger_auc,
            "folds": folds.to_dict("records"),
            "train_rows": len(X),
            "feature_importances": importances,
        })
        self.journal.log_model_event(self.model_name, version, "trained",
                                     {"auc": challenger_auc})

        if champion_auc is None or challenger_auc >= champion_auc:
            self.registry.promote(self.model_name, version)
            self.journal.log_model_event(self.model_name, version, "promoted",
                                         {"auc": challenger_auc,
                                          "beat": champion_auc})
            return version
        logger.info("challenger AUC {:.4f} < champion {:.4f}; keeping champion",
                    challenger_auc, champion_auc)
        self.journal.log_model_event(self.model_name, version, "rejected",
                                     {"auc": challenger_auc,
                                      "champion_auc": champion_auc})
        return None

    # ----------------------------------------------------------- drift
    def check_drift(self, live_returns: pd.Series,
                    current_importances: Optional[pd.Series] = None) -> bool:
        """
        Performance drift: rolling live Sharpe below the configured floor.
        Feature drift: rank correlation of importances vs training-time.
        Rolls back (if configured) and returns True when drift is detected.
        """
        from quant.metrics.performance import sharpe

        drifted = False
        if len(live_returns.dropna()) >= 20:
            live_sharpe = sharpe(live_returns)
            if live_sharpe < config.DRIFT_SHARPE_FLOOR:
                logger.warning("performance drift: live Sharpe {:.2f} < floor {:.2f}",
                               live_sharpe, config.DRIFT_SHARPE_FLOOR)
                drifted = True

        if current_importances is not None and not current_importances.empty:
            try:
                _, meta = self.registry.load(self.model_name)
                trained = pd.Series(meta.get("feature_importances", {}))
                common = trained.index.intersection(current_importances.index)
                if len(common) >= 5:
                    rank_corr = trained[common].rank().corr(
                        current_importances[common].rank())
                    if rank_corr < 0.5:
                        logger.warning("feature-importance drift: rank corr {:.2f}",
                                       rank_corr)
                        drifted = True
            except FileNotFoundError:
                pass

        if drifted and config.ROLLBACK_ON_UNDERPERFORM:
            rolled = self.registry.rollback(self.model_name)
            if rolled:
                self.journal.log_model_event(self.model_name, rolled,
                                             "rollback_on_drift")
        return drifted
