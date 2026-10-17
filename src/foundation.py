"""Foundation-model forecasters: TimesFM (local, on CPU) and TimeGPT (Nixtla's API).

Both are zero-shot: 28 days of history in, one day (96 steps) out. They use the
same signature as the forecasters in forecast.py, and also have forecast_days(),
which runs a whole year in batches and caches every (model, date) under
out/cache/, so reruns are instant and nothing needs the checkpoint or a key.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import FREQ, INTERVALS_PER_DAY, fingerprint
from .forecast import check_history, history_before, target_index_for

log = logging.getLogger("peakshave.foundation")


class ZeroShotForecaster:
    """Base class for the foundation models: batching + the forecast cache.

    The cache is one Parquet file per model and dataset,
    out/cache/<model>/<data fingerprint>.parquet, with a row per 15-minute step
    (kw = median, q10, q90). Forecasts are looked up by date inside it. The
    fingerprint is a hash of the load data, so changing the data can't
    silently reuse old forecasts.
    """

    name = "zero-shot"
    batch_size = 32

    def __init__(self, context_days: int = config.FM_CONTEXT_DAYS, cache_root: Path = config.CACHE):
        self.context_days = context_days
        self.cache_root = cache_root
        self.fingerprint: str | None = None
        self._cache = None  # (fingerprint, frame) once read from disk

    def __call__(self, history: pd.Series, target_idx: pd.DatetimeIndex) -> pd.Series:
        # single-day version with the usual signature
        check_history(history, target_idx)
        day = target_idx[0].normalize()
        if day in self._cached_days():
            return self._cached()["kw"].reindex(target_index_for(day))
        out = self._predict([self._context(history)], [day])
        if out is None:
            raise RuntimeError(f"{self.name} is unavailable and has no cached forecast for {day.date()}")
        self._save(out)
        return out[0]["kw"]

    def forecast_days(self, load: pd.Series, days: pd.DatetimeIndex) -> pd.Series | None:
        self.fingerprint = fingerprint(load)
        have_days = self._cached_days()
        missing = []
        for d in days:
            if d not in have_days:
                missing.append(d)

        if missing and self._load() is not None:
            log.info("%s: %d/%d days not cached, running model", self.name, len(missing), len(days))
            t0 = time.time()
            for i in range(0, len(missing), self.batch_size):
                chunk = missing[i : i + self.batch_size]
                contexts = []
                for d in chunk:
                    contexts.append(self._context(history_before(load, d)))
                out = self._predict(contexts, chunk)
                if out is None:
                    break
                self._save(out)  # after every batch, so a long run can resume
                done = min(i + self.batch_size, len(missing))
                log.info("%s: %d/%d days done (%.0fs)", self.name, done, len(missing), time.time() - t0)

        have_days = self._cached_days()
        have = 0
        stamps = []
        for d in days:
            if d in have_days:
                have += 1
            stamps.append(target_index_for(d).to_numpy())
        if have < len(days):
            log.warning("%s: only %d/%d days available; leaving it out of the comparison", self.name, have, len(days))
            return None
        index = pd.DatetimeIndex(np.concatenate(stamps))
        return self._cached()["kw"].reindex(index).rename("kw")

    def _context(self, history: pd.Series) -> pd.Series:
        return history.iloc[-self.context_days * INTERVALS_PER_DAY :]

    def _cache_path(self) -> Path:
        return self.cache_root / self.name / f"{self.fingerprint or 'unknown'}.parquet"

    def _cached(self) -> pd.DataFrame:
        # everything this model has forecast for this data, read once
        if self._cache is None or self._cache[0] != self.fingerprint:
            path = self._cache_path()
            if path.exists():
                frame = pd.read_parquet(path)
            else:
                frame = pd.DataFrame({"kw": [], "q10": [], "q90": []}, index=pd.DatetimeIndex([], name="timestamp"))
            self._cache = (self.fingerprint, frame)
        return self._cache[1]

    def _cached_days(self) -> set:
        return set(self._cached().index.normalize().unique())

    def _save(self, frames: list[pd.DataFrame]) -> None:
        # add the new days and write the whole file back; it's only a few MB at most
        parts = []
        if len(self._cached()):
            parts.append(self._cached())
        for f in frames:
            parts.append(f)
        merged = pd.concat(parts)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index().round(2)
        merged.index.name = "timestamp"
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path)
        self._cache = (self.fingerprint, merged)

    def _load(self):
        # subclasses return the model/client, or None if it can't run here
        return self

    def _predict(self, contexts: list[pd.Series], days: list[pd.Timestamp]) -> list[pd.DataFrame] | None:
        raise NotImplementedError


class TimesFMZeroShot(ZeroShotForecaster):
    """Google's TimesFM, run locally on CPU. 28 days of context in, 96 steps out.

    pip install "timesfm[torch]". First run downloads the checkpoint from
    Hugging Face (~1.3 GB for 3.0). The 3.0 weights are non-commercial only;
    2.5 is Apache-2.0, which is why both versions are supported.
    """

    REPOS = {"3.0": "google/timesfm-3.0-pytorch", "2.5": "google/timesfm-2.5-200m-pytorch"}

    def __init__(self, version: str = "3.0", **kw):
        super().__init__(**kw)
        self.version = version
        self.name = f"timesfm-{version}"
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
        except ImportError:
            log.warning("timesfm: torch not installed; run `pip install \"timesfm[torch]\"`")
            return None
        torch.set_num_threads(os.cpu_count() or 4)
        try:
            if self.version == "3.0":
                from timesfm3 import TimesFM3Forecaster

                self._model = TimesFM3Forecaster.from_pretrained(
                    self.REPOS["3.0"], device="cpu", per_core_batch_size=self.batch_size
                )
            else:
                import timesfm

                # torch.compile wants a C++ compiler on Windows, and eager mode is fine here
                model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.REPOS["2.5"], torch_compile=False)
                model.compile(
                    timesfm.ForecastConfig(
                        max_context=self.context_days * INTERVALS_PER_DAY,
                        max_horizon=128,
                        normalize_inputs=True,
                        use_continuous_quantile_head=True,
                        infer_is_positive=True,
                        fix_quantile_crossing=True,
                        per_core_batch_size=self.batch_size,
                    )
                )
                self._model = model
        except ImportError:
            log.warning("timesfm: package not installed; run `pip install \"timesfm[torch]\"`")
            return None
        return self._model

    def _predict(self, contexts, days):
        model = self._load()
        if model is None:
            return None
        arrays = []
        for c in contexts:
            arrays.append(c.to_numpy(dtype=np.float32))

        point, q10, q90 = [], [], []
        if self.version == "3.0":
            outs = model.predict_batch(arrays, horizon=INTERVALS_PER_DAY, return_quantiles=True, make_positive=True)
            for o in outs:
                point.append(o.forecast)
                q10.append(o.quantiles[:, 0])  # quantiles are 0.1 ... 0.9
                q90.append(o.quantiles[:, -1])
        else:
            # 2.5 pads the list it's given up to a full batch, in place, so hand it a copy
            point_arr, quant_arr = model.forecast(horizon=INTERVALS_PER_DAY, inputs=list(arrays))
            # it returns the mean first, then the deciles 0.1 ... 0.9
            for i in range(len(contexts)):
                point.append(point_arr[i])
                q10.append(quant_arr[i, :, 1])
                q90.append(quant_arr[i, :, 9])

        frames = []
        for i, d in enumerate(days):
            frames.append(pd.DataFrame({"kw": point[i], "q10": q10[i], "q90": q90[i]}, index=target_index_for(d)))
        return frames


class TimeGPTZeroShot(ZeroShotForecaster):
    """Nixtla's TimeGPT through their API. Same 28 days in, 96 steps out.

    Needs NIXTLA_API_KEY (environment or .env). Without it we log a warning
    and only use whatever is already cached.
    """

    def __init__(self, model: str | None = None, **kw):
        super().__init__(**kw)
        self.model = model or os.environ.get("TIMEGPT_MODEL", "timegpt-2.1")
        self.name = self.model
        self._client = None
        self._warned = False

    def _load(self):
        if self._client is not None:
            return self._client
        config.load_dotenv()
        if not os.environ.get("NIXTLA_API_KEY"):
            if not self._warned:
                log.warning("timegpt: NIXTLA_API_KEY not set; skipping live calls (cached forecasts still used)")
                self._warned = True
            return None
        try:
            from nixtla import NixtlaClient
        except ImportError:
            log.warning("timegpt: nixtla not installed; run `pip install nixtla`")
            return None
        self._client = NixtlaClient()  # reads NIXTLA_API_KEY itself
        return self._client

    def _predict(self, contexts, days):
        client = self._load()
        if client is None:
            return None
        # each day's context goes in as its own series, so one request covers a batch of days
        frames = []
        for ctx, d in zip(contexts, days):
            frames.append(pd.DataFrame({"unique_id": f"{d:%Y-%m-%d}", "ds": ctx.index, "y": ctx.to_numpy()}))
        df = pd.concat(frames, ignore_index=True)
        fc = client.forecast(
            df=df, h=INTERVALS_PER_DAY, freq=FREQ, model=self.model, level=[80],
            id_col="unique_id", time_col="ds", target_col="y",
        )

        lo_col = hi_col = point_col = None
        for c in fc.columns:
            if c.endswith("lo-80"):
                lo_col = c
            elif c.endswith("hi-80"):
                hi_col = c
            elif c not in ("unique_id", "ds") and point_col is None:
                point_col = c

        out = []
        for d in days:
            part = fc[fc["unique_id"] == f"{d:%Y-%m-%d}"].sort_values("ds")
            idx = target_index_for(d)
            if len(part) != len(idx):
                raise RuntimeError(f"TimeGPT returned {len(part)} steps for {d.date()}, expected {len(idx)}")
            out.append(
                pd.DataFrame(
                    {
                        "kw": part[point_col].to_numpy(),
                        "q10": part[lo_col].to_numpy() if lo_col else np.nan,
                        "q90": part[hi_col].to_numpy() if hi_col else np.nan,
                    },
                    index=idx,
                )
            )
        return out


_FM_INSTANCES: dict[str, ZeroShotForecaster] = {}


def timesfm_zeroshot(history: pd.Series, target_index: pd.DatetimeIndex, version: str = "3.0") -> pd.Series:
    # same signature as the others; for a whole year use TimesFMZeroShot().forecast_days
    key = f"timesfm-{version}"
    if key not in _FM_INSTANCES:
        _FM_INSTANCES[key] = TimesFMZeroShot(version)
    model = _FM_INSTANCES[key]
    if model.fingerprint is None:
        model.fingerprint = fingerprint(history)
    return model(history, target_index)


def timegpt_zeroshot(history: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    if "timegpt" not in _FM_INSTANCES:
        _FM_INSTANCES["timegpt"] = TimeGPTZeroShot()
    model = _FM_INSTANCES["timegpt"]
    if model.fingerprint is None:
        model.fingerprint = fingerprint(history)
    return model(history, target_index)
