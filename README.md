# BiAB-IoT: Bidirectional AI-Blockchain Security Architecture for IoT

Reference implementation for the manuscript
**"A Bidirectional AI-Blockchain Security Architecture for IoT with Cross-Layer Information Exchange"**
by Angel Hepzibah R and Roselin J

\---

## 1\. Description

This repository provides the Python / TensorFlow implementation used to produce the
experimental results reported in the manuscript. BiAB-IoT is a three-tier security
framework for IoT that couples:

* **Tier 1 — Device Layer:** simulated IoT devices generating network-flow telemetry.
* **Tier 2 — AI Layer:** an Isolation-Forest / neural-network anomaly detector combined
with a *reputation-weighted* federated averaging engine (mechanism **P2**,
Blockchain → AI).
* **Tier 3 — Blockchain Layer:** a PBFT-consensus policy layer driven by
*AI-generated* policy proposals (mechanism **P1**, AI → Blockchain).

The two mechanisms share a single on-chain reputation register, so a P1 penalty is
immediately visible to P2 aggregation in the same operational cycle.

The seven notebooks in this repository reproduce, in order:

|Notebook|Purpose|
|-|-|
|`BIAB.ipynb`|Data preparation from CIC-IoT-2023 CSVs, baseline federated learning (10 rounds).|
|`BIAB2.ipynb`|Adversarial scenario: 10 % of clients perform label-flipping poisoning.|
|`BIAB3.ipynb`|BiAB-IoT defence with statistical (attack-ratio) detection of poisoned clients.|
|`Notebook\_A\_Baselines.ipynb`|B1 Centralised Isolation Forest, B3 FL + local anomaly filtering, B4 FLIT reputation-weighted FL.|
|`Notebook\_B\_BiAB\_Sweep.ipynb`|BiAB-IoT (P1+P2) attack-percentage sweep (0/5/10/15/20 % malicious) + P1/P2 ablation study.|
|`Notebook\_C\_PBFT\_Benchmark.ipynb`|Real ECDSA P-256 microbenchmark + PBFT three-phase simulation across k=4 and k=7 validators.|
|`Notebook\_D\_Aggregate.ipynb`|Aggregates all result pickles from A/B/C into Tables 2-6 (CSV) and Figures 3-6 (PNG).|

Together they reproduce all numerical results in Tables 2 through 6 of the manuscript.

\---

## 2\. Dataset Information

**Name:** CIC-IoT-2023 (also written CICIoT2023)
**Provider:** Canadian Institute for Cybersecurity (CIC), University of New Brunswick.
**Landing page / download:** [https://www.unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
**Primary citation:**

> E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu and A. A. Ghorbani.
> "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT
> Environment," \*Sensors\*, vol. 23, no. 13, 5941, 2023.
> DOI: <https://doi.org/10.3390/s23135941>

**Content used in this work:** the per-flow CSV feature files distributed under the
`CSV/` subdirectory of the dataset. Each row is a network flow with 46 numeric
features and a `label` column (benign traffic or one of 33 attack types). Labels
are binarised in the pipeline: `BenignTraffic` → 0, all other classes → 1.

**Subset used:** 10 CSV files sampled at random with seed 42 from the full CSV
directory (module 4 of `BIAB.ipynb`). The resulting frame is train/test split
75 / 25 with a stratified split (module 7).

**Licence:** the dataset is released by CIC for research use under the terms
stated on the landing page above. It is **not redistributed here** — users must
download it directly from CIC.

\---

## 3\. Code Information

The pipeline is organised as three Jupyter notebooks. All are self-contained and
run top-to-bottom.

### 3.1 `BIAB.ipynb` — data preparation \& baseline FL

Modules 1–20:

|Module|Function|
|-|-|
|1|Mount Google Drive (Colab environment).|
|2|Import NumPy, pandas, scikit-learn, TensorFlow / Keras.|
|3–4|Locate and randomly sample 10 CSV files from `/dataset`.|
|5|Build binary label (`0` = benign, `1` = attack); print class distribution.|
|6|Feature / label separation (`X`, `y`).|
|7|Stratified 75/25 train / test split (`random\_state=42`).|
|8|Standardisation with `StandardScaler` (fit on train).|
|9–11|Persist scaler and pickled dataset to Drive.|
|12|Partition `X\_train` into 500 IID clients (`NUM\_CLIENTS = 500`).|
|13|Save `client\_data.pkl`.|
|14|Define the global model — Keras `Sequential(256/128/64/1)` with sigmoid head.|
|15–16|`train\_client()` and `federated\_average()` helpers.|
|17|Baseline federated learning: 10 rounds × 50 sampled clients per round.|
|18|Poisoning setup — 10 % of clients (50 devices) selected as malicious.|
|19|Produce `poisoned\_client\_data` with label-flipped `y`.|
|20|Final evaluation of the baseline model.|

### 3.2 `BIAB2.ipynb` — adversarial baseline (attack, no defence)

Loads `binary\_dataset.pkl` produced by `BIAB.ipynb`, executes 10 federated
rounds *with* the poisoned client data active and *no* mitigation, and reports
accuracy, precision, recall, F1, and confusion matrix.

### 3.3 `BIAB3.ipynb` — BiAB-IoT defence

Loads the same pickle and applies statistical detection of poisoned clients
as an experimental analogue of the manuscript's Algorithm 2 (reputation-based
exclusion):

* The defender computes the local attack-ratio `mean(y\_client)` for every
client. Clients whose ratio falls outside the plausible band `\[0.05, 0.95]`
are flagged as suspect (the experimental analogue of a low on-chain
reputation triggering a policy proposal).
* Flagged clients are excluded from subsequent aggregation rounds. **The
ground-truth `malicious\_ids` set is NOT used to make the exclusion
decision** — the filter operates only on the statistical signal available to
the defender at run time. `malicious\_ids` is used only in the diagnostic
print-out at the end of the cell to report true positives, false positives,
and false negatives of the detection step.
* Ten federated rounds are executed on the trusted subset only.

Final metrics (accuracy / precision / recall / F1 / confusion matrix) are
reported and compared with the baseline and attacked versions in the last cell.

\---

## 4\. Usage Instructions

The notebooks were developed in Google Colab (GPU runtime not required). They
can be run identically on any local Jupyter / JupyterLab install.

**Local run**

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the CIC-IoT-2023 CSV files
#    from https://www.unb.ca/cic/datasets/iotdataset-2023.html
#    and place them in ./dataset/

# 4. Open the notebooks
jupyter notebook BIAB.ipynb        # produces binary\_dataset.pkl
jupyter notebook BIAB2.ipynb       # attacked federated model
jupyter notebook BIAB3.ipynb       # BiAB-IoT defended federated model
```

**Colab run**

Upload the three notebooks and the CIC-IoT-2023 `CSV/` folder to
`/content/drive/MyDrive/dataset/`. Run the cells top-to-bottom in the order
above; the pickled intermediate artefacts are shared through the Drive folder.

**Reproducing the reported numbers**

Random seeds are fixed to `42` in all three notebooks (`random.seed(42)` and
`train\_test\_split(..., random\_state=42)`). Running the notebooks in the same
order on the same 10-CSV sample reproduces the numbers reported in Tables 2–4
of the manuscript within ±0.5 percentage points.

\---

## 5\. Requirements

|Package|Version tested|
|-|-|
|Python|≥ 3.9|
|numpy|1.26.x|
|pandas|2.1.x|
|scikit-learn|1.4.x|
|tensorflow|2.15.x|
|matplotlib|3.8.x|

A minimal `requirements.txt` is included in the same directory.

Hardware: any modern CPU is sufficient (the reference dense network is
\~50 k parameters). A GPU is not required.

\---

## 6\. Methodology

The overall training / evaluation pipeline is:

1. **Data ingestion.** Ten randomly-sampled CIC-IoT-2023 CSVs (seed 42) are
concatenated into a single frame.
2. **Binary labelling.** `BenignTraffic` → 0; every other class → 1.
3. **Stratified split.** 75 % train / 25 % test.
4. **Feature scaling.** Zero-mean unit-variance via `StandardScaler`.
5. **Client partitioning.** IID partition into `NUM\_CLIENTS = 500` shards.
6. **Baseline federated learning** (`BIAB.ipynb`). 10 rounds × 50 sampled
clients × 1 local epoch × batch 64. `federated\_average()` implements
`FedAvg`.
7. **Poisoning simulation** (`BIAB2.ipynb`). 10 % of clients invert their
labels (`y\_poisoned = 1 - y`). Same 10-round schedule.
8. **BiAB-IoT defence** (`BIAB3.ipynb`). Attack ratio per client is compared
against thresholds `\[0.05, 0.95]`; clients outside this range are flagged
and excluded — the experimental analogue of a P1 quarantine transaction on
the shared on-chain reputation register (Section 5 of the manuscript). The
remaining `trusted\_clients` set participates in 10 federated rounds of
secure aggregation.
9. **Evaluation.** Accuracy, precision, recall, F1, confusion matrix on the
held-out test set.

The blockchain layer (PBFT consensus, ECDSA P-256 signing, on-chain reputation
register) is modelled analytically at the client-selection level in this
reference code; the cryptographic overhead numbers reported in Table 5 of the
manuscript come from separate ECDSA measurement runs and are described
inline in the manuscript's Experimental Setup section.

