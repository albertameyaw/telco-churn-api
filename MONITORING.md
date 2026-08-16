# Monitoring the Churn Model in Production

A model that scores well in `python -m src.train` today doesn't stay
good forever. This document covers why models degrade, how to detect it
(with and without labels), what to actually monitor operationally, what
tools to use, and how to decide when to retrain.

## Why models degrade in production

A model learns a fixed snapshot of the relationship between features and
the target at training time. Production degrades that relationship in a
few distinct ways:

- **The world changes.** New competitors, price changes, a pandemic, a
  new phone plan tier - customer behavior shifts and the patterns the
  model learned stop being true.
- **The input distribution shifts.** Even if the underlying relationship
  is stable, the *mix* of customers changes - e.g. a marketing push
  brings in a wave of month-to-month customers, who behave differently
  than the training population's average customer.
- **Upstream data changes silently.** A field gets renamed, a system
  starts sending `null` where it used to send `0`, a currency changes -
  these are engineering failures that look identical to the model as
  "the customers changed."
- **Feedback loops.** If retention offers are triggered by this model's
  predictions, the model is now influencing the very behavior it's
  trying to predict, which can bias future training data.

## Three types of drift

These are easy to conflate but the detection method and the fix differ
for each.

- **Data drift (a.k.a. covariate shift):** the distribution of *input
  features* changes, independent of whether the relationship to the
  target changed. Example: average `tenure` in incoming requests drops
  because of a new-customer acquisition campaign.
- **Concept drift:** the relationship between features and the target
  changes - the same input now implies a different outcome. Example: a
  `Contract=Month-to-month` customer used to churn at 40%, now churns at
  60% because a competitor launched a cheaper plan.
- **Prediction drift:** the distribution of the model's *output*
  (predicted probabilities or predicted classes) changes over time. This
  is often the earliest and cheapest signal to monitor, because it's a
  downstream symptom of either data or concept drift - you don't need
  ground truth to compute it, just the logged predictions themselves.

Rule of thumb: prediction drift tells you *something* changed; data
drift narrows it to *inputs*; concept drift confirms the *model's
learned relationship* is stale (but concept drift is the hardest to
detect directly, since it requires labels).

## Detecting drift without ground-truth labels

Churn labels arrive late - you only know if a customer churned after
enough time has passed. Until then, you can still monitor the inputs and
outputs you already have.

- **PSI (Population Stability Index):** buckets a feature's values into
  bins, compares the % of the current window's data in each bin against
  the training distribution, and sums a weighted log-ratio across bins.
  A single number per feature. Common thresholds: <0.1 stable, 0.1-0.25
  moderate shift worth watching, >0.25 significant shift. Works well on
  both numeric (after binning) and categorical features, and on the
  model's output probabilities themselves (prediction drift).
- **KS test (Kolmogorov-Smirnov):** a statistical test comparing two
  *continuous* distributions (e.g. `MonthlyCharges` this week vs.
  `MonthlyCharges` in training data) by the maximum gap between their
  cumulative distributions. Gives a p-value - useful for numeric
  features like `tenure`, `MonthlyCharges`, `TotalCharges`.
  *Concept to read up on: what a p-value actually means* - it's easy to
  over-interpret ("p<0.05 means definitely drifted") when at high enough
  sample sizes even trivial, harmless shifts become "significant."
  Pair the test with a practical-significance threshold, not just p<0.05.
- **Chi-square test:** the categorical-feature analog of KS - compares
  observed vs. expected frequency counts across categories (e.g. has the
  mix of `Contract` types shifted?).

Practically: run these weekly (or per-batch) per feature, log the
scores, and alert when they cross a threshold - this is monitoring the
model's *inputs*, catching data drift before it necessarily shows up in
outcomes.

## Detecting drift with labels, once they arrive

Once actual churn outcomes are known (e.g. 30-90 days later, whenever
"churn" is operationally defined), compute the same metrics used in
training - but on a rolling window of recent predictions instead of a
static test set:

- **Rolling accuracy/precision/recall/F1/ROC-AUC** over the trailing N
  days or last N predictions, plotted as a time series. A steady decline
  is the clearest signal that concept drift (not just data drift) is
  actually hurting real-world performance.
- **Compare rolling metrics against the training-time baseline.** Define
  an acceptable degradation band (e.g. "alert if rolling ROC-AUC drops
  more than 0.05 below the training value") rather than an absolute
  floor, since some seasonal wobble is normal.
- **Segment the rolling metrics** by key slices (contract type,
  tenure bucket, region if available) - an aggregate metric can look
  fine while one segment silently degrades, especially if that segment
  is a small share of volume.

## Operational monitoring

Separate from *model quality* monitoring - this is *service health*,
and it matters even if the model itself is perfectly accurate:

- **Latency:** p50/p95/p99 response time for `/predict`. A slow model
  (or a slow upstream feature lookup) degrades UX even with correct
  predictions.
- **Error rate:** 4xx (bad input - e.g. Pydantic validation failures,
  worth tracking as an early data-drift signal too) vs. 5xx (server-side
  failures) tracked separately, since they imply different fixes.
- **Throughput:** requests/second, to catch both unexpected traffic
  spikes and unexpected drop-offs (a drop-off can mean an upstream
  caller broke, which is just as worth knowing as a spike).

## Tooling suggestions

- **Evidently AI:** open-source Python library purpose-built for ML
  monitoring - computes PSI/KS/chi-square drift reports and data quality
  checks out of the box, and can generate scheduled HTML/JSON reports or
  feed a dashboard. Good fit here since it works directly on pandas
  DataFrames, same as this training pipeline.
- **Prometheus + Grafana:** standard for the *operational* metrics
  (latency, error rate, throughput) - instrument the FastAPI app with
  `prometheus-fastapi-instrumentator` (or hand-rolled middleware) to
  expose a `/metrics` endpoint Prometheus scrapes, then build Grafana
  dashboards and alerts on top. This is the same pattern you'd use for
  monitoring a Spring Boot service with Micrometer + Prometheus - same
  idea, different library.
- **Prediction logging:** log every request's input features, the
  prediction, the probability, a timestamp, and (once available) the
  actual outcome, to a durable store (a database table, or an
  append-only file/data lake). This log is the raw material every drift
  and rolling-metric calculation above depends on - without it, none of
  the above is possible after the fact.

## Retraining strategy

- **Scheduled retraining:** retrain on a fixed cadence (e.g. monthly)
  regardless of detected drift, on the assumption that customer behavior
  and offers change often enough that periodic refresh is cheap
  insurance. Simple to operate, but can retrain unnecessarily or miss a
  fast drift event that happens between scheduled runs.
- **Threshold-triggered retraining:** retrain when a monitored metric
  (PSI above threshold, rolling ROC-AUC below the acceptable band) fires
  an alert. More responsive and avoids wasted retraining, but needs the
  monitoring pipeline above to already be reliable - it's only as good
  as the alerts triggering it.
- **In practice, combine both:** scheduled retraining as a baseline
  cadence, with threshold-triggered retraining as an early-warning
  override for faster-than-expected drift.
- **Shadow deployment before promotion:** once a candidate model is
  retrained, don't replace the production model directly. Run the new
  model *alongside* the current one on live traffic - it receives the
  same requests and logs its predictions, but the old model's output is
  still what's actually returned to callers. Compare the candidate's
  logged predictions/metrics against the incumbent over a defined
  window; only promote (swap which model's output is actually served)
  if the candidate is clearly better or at least not worse. This avoids
  regressing production quality based on a retraining run that looked
  good on a held-out test set but doesn't generalize to current live
  traffic.

## Applying this to this specific project

For this take-home's scope, the most valuable minimal setup would be:

1. Log every `/predict` request/response (input, prediction, probability,
   timestamp) to a file or table - this alone unlocks everything else
   in this document.
2. A scheduled (e.g. weekly) job that runs Evidently's drift report
   comparing the past week's logged inputs against the training data
   distribution.
3. Prometheus counters/histograms on the FastAPI app for latency,
   status codes, and request volume, with a basic Grafana dashboard.
4. Once enough labels accumulate, a rolling ROC-AUC/F1 chart compared
   against the training-time baseline, with an alert threshold that
   triggers a retraining review (not necessarily automatic retraining).
