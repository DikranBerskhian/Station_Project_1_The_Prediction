# Station Project 1: The Prediction

## Introduction

This TUMO Labs project chooses **Online Payments Fraud Detection**. It trains a supervised binary classifier to predict `isFraud` from payment information available before a transaction finishes. The deliverable is `prediction.ipynb`, with 24 numbered Markdown/code cells, beginner-friendly explanations and visible outputs. It replaces the original notebook that failed because KaggleHub was missing.

The implementation follows the twelve requirements supplied in the request. The original assignment attachment was not available in the workspace, so any additional requirements in that document still need to be checked manually.

## Dataset and inputs

Source: [Synthetic Financial Datasets for Fraud Detection / PaySim, ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1). PaySim is synthetic mobile money transaction data, not real customer records. The notebook calls `kagglehub.dataset_download("ealaxi/paysim1")`, searches the returned folder recursively and loads the unique CSV with the expected schema using `pd.read_csv()`. It does not depend on a username, absolute path or a preexisting workspace CSV. An ambiguous/missing CSV produces an explicit error.

| Input | Purpose |
|---|---|
| `step` | Simulation time step, measured in hours |
| `amount` | Payment amount |
| `oldbalanceOrg` | Origin account balance before the payment |
| `oldbalanceDest` | Destination account balance before the payment |
| `type` | CASH_IN, CASH_OUT, DEBIT, PAYMENT or TRANSFER; encoded as binary columns |
| **Target: `isFraud`** | **0 = Not fraud; 1 = Fraud** |

Excluded inputs:

- `nameOrig`, `nameDest`: customer identifiers can encourage customer memorization and are unsuitable as raw numeric/categorical predictors for this exercise.
- `isFlaggedFraud`: an existing rule-based flag; excluding it measures our detector independently of that rule.
- `newbalanceOrig`, `newbalanceDest`: balances recorded after the transaction. Using them for a decision before a transaction finishes would use unavailable outcome information. A post-transaction investigation model could use them under different assumptions.

Zero balances are retained. They can be valid, and some destination balance information is unavailable in the simulation. Even the old destination balance may be unavailable in a real deployment.

## Installation

Use a supported Python 3 environment; the actual execution report below identifies the version used here. Open a terminal in this project folder. A separate virtual environment is recommended:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name tumo-prediction --display-name "TUMO Prediction"
.\.venv\Scripts\python.exe -m jupyterlab
```

Open `prediction.ipynb`, select **TUMO Prediction**, and run **Kernel → Restart Kernel and Run All Cells**. The notebook also has a `%pip install` cell containing all required packages. If installation requests a kernel restart, restart and run all cells again. The package table records the versions used. On another operating system, use that virtual environment's Python executable in place of the Windows path.

The first run requires internet access, disk space for a roughly 0.5 GB CSV and its download, and several GB of available memory for the full dataset and training. KaggleHub caches downloaded data. If Kaggle requests authentication, follow the [official KaggleHub instructions](https://github.com/Kaggle/kagglehub#authentication); do not put credentials into the notebook or commit them. Network/authentication failures are surfaced rather than silently switching datasets.

## Running and testing

Run from this project folder so local results appear in `outputs/`. Execute all cells in order; later cells depend on earlier ones. The final code cell prints **Run checks passed** only after verifying the split sizes, nonoverlapping row indices, binary predictions, confusion matrix total, accuracy calculation, baseline recall and saved JSON contents. Inspect the charts and tables as well. The checks verify implementation consistency, not production reliability.

An optional command executes the notebook from a fresh kernel and saves its outputs:

```powershell
python run_notebook.py
```

This helper uses the Jupyter kernel named `python3`. Run it with the intended Python environment and ensure that kernel points to that environment. For example, to register the activated environment as that kernel:

```powershell
python -m ipykernel install --user --name python3 --display-name "Python 3"
python run_notebook.py
```

The helper stops on errors and preserves partial notebook outputs for diagnosis. A successful run prints **All code cells executed successfully; notebook outputs saved.** Save the notebook after an interactive run too, so reviewers can see its outputs without downloading the data themselves. Re-running overwrites the result files.

## Workflow and outputs at each major step

| Notebook cells | Operation | Visible output |
|---|---|---|
| 01–05 | Introduction, installation, imports | Installation log and package-version table |
| 06–07 | KaggleHub download and Pandas load | Cache path, selected CSV filename, `(rows, columns)` and first five rows |
| 08–09 | Inspection | Reproducible sample, shape, dtype/missing-value table, exact duplicate count, class/type count tables |
| 10–11 | Visualization | Interactive transaction-type pie and fraud/non-fraud pie; computed fraud percentage |
| 12–13 | Features and target | Five raw input columns and target counts after exact duplicate removal |
| 14–15 | Stratified split | Training/test shapes and class counts in each partition |
| 16–17 | Preprocessing and training | Training completion, tree depth/leaves and encoded feature names |
| 18–19 | Evaluation | Accuracy/precision/recall/F1 comparison, both confusion tables, tree heatmap and interpreted fraud counts |
| 20–21 | Illustrative prediction | One-row input table and the predicted text **Fraud** or **Not fraud** |
| 22–23 | Local export and checks | Saved paths, JSON contents, comparison CSV table and check confirmation |
| 24 | Limitations/submission | Requirement checklist and remaining manual steps |

The execution report below records actual output examples and results from this workspace. The notebook contains the sample records and complete tables/charts.

## Preprocessing and training justification

The notebook counts and removes exact full-row duplicates before splitting. It rejects missing/invalid targets, since a target cannot be imputed responsibly. It does not drop legitimate zero balances or delete transactions merely because their selected input values match.

`train_test_split(test_size=0.20, random_state=42, stratify=y)` reserves 20% for testing. Stratification preserves the rare fraud label approximately in both partitions. A `Pipeline` combines a `ColumnTransformer` and the decision tree. Numeric predictors use `SimpleImputer(strategy="median")`; transaction types use the training most frequent value and `OneHotEncoder(handle_unknown="ignore")`. Both imputers and the encoder are fitted **only on training data** by `model.fit(X_train, y_train)`. Test data and new examples pass through these fitted transformations. No scaling is needed for the tree. See the [official guidance on leakage and consistent preprocessing](https://scikit-learn.org/stable/common_pitfalls.html).

A `DecisionTreeClassifier` learns if/else rules by greedily splitting labeled data to reduce weighted Gini impurity. It is a readable educational starting point for nonlinear tabular patterns. Parameters were fixed before the test evaluation:

| Parameter | Value | Reason |
|---|---|---|
| `criterion` | `gini` | Prefer purer child nodes |
| `max_depth` | `8` | Limit complexity and overfitting |
| `min_samples_leaf` | `20` | Discourage very small leaves |
| `class_weight` | `balanced` | Weight rare fraud more heavily using training frequencies; can increase false alarms |
| `random_state` | `42` | Reproducible randomness |

These choices are not claimed to be optimal. The test set is used for final evaluation, never for parameter/threshold selection. Future tuning must use a validation subset or cross-validation within training data and then a fresh final evaluation. See [DecisionTreeClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

## Evaluation and example prediction

`model.score(X_test, y_test)` explicitly measures accuracy. Accuracy can look excellent when almost every transaction is legitimate, even if every fraud is missed. The comparison uses `DummyClassifier(strategy="constant", constant=0)`, an always-non-fraud baseline. Both models use the same held-out labels.

Precision measures the fraction of fraud alerts that are correct; recall measures the fraction of actual fraud caught; F1 balances these two. All three treat fraud (`1`) as positive. The baseline's undefined precision is reported as zero using `zero_division=0`. Confusion matrices are ordered `[0, 1]`, with rows = actual and columns = predicted: `[[TN, FP], [FN, TP]]`. False positives are legitimate payments flagged as fraud; false negatives are frauds allowed through.

The example is explicitly **illustrative**, since no user transaction was supplied. Its five inputs are `step=100`, `amount=200000.0`, `oldbalanceOrg=200000.0`, `oldbalanceDest=0.0`, `type="TRANSFER"`. It has no known true label. A one-row DataFrame uses exactly the training input columns and their order. `model.predict(example)` reuses the fitted pipeline, and the output is displayed as **Fraud** or **Not fraud**. Actual results appear below and in the saved notebook.

<!-- EXECUTION_REPORT_START -->
### Actual execution report (2026-11-02)

**Executed successfully from top to bottom in a fresh kernel.** All 11 code cells completed, the final checks passed, three Plotly figure outputs were saved (two pies and one confusion heatmap), and all three result files were created. No metric or prediction below is invented. The Windows runner emitted an event-loop compatibility warning; it did not prevent execution.

Observed output examples:

- Environment: Python **3.14.0**; kagglehub 1.0.3, pandas 3.0.2, numpy 2.4.3, plotly 6.7.0, scikit-learn 1.9.1.
- Download/load: KaggleHub downloaded `ealaxi/paysim1`; CSV `PS_20174392719_1491204439457_log.csv`; shape **(6,362,620, 11)**. The downloaded cache path is printed in cell 07.
- Inspection: **0 missing values in every column**, **0 exact duplicate rows**, **6,354,407 Not fraud** and **8,213 Fraud** records. Cells 07 and 09 show the actual sample rows and dtype table.
- Visualization: fraud is **0.129082%** of all records. The type pie uses CASH_OUT **2,237,500**, PAYMENT **2,151,495**, CASH_IN **1,399,284**, TRANSFER **532,909** and DEBIT **41,432**.
- Features/split: **6,362,620** modeling rows with **5 raw features**; training shape **(5,090,096, 5)**, test shape **(1,272,524, 5)**. Training contains **6,570 frauds / 5,083,526 non-frauds**; testing contains **1,643 frauds / 1,270,881 non-frauds**.
- Training: **depth 8**, **67 leaves**, **9 transformed columns**: four numeric predictors plus one binary column for each of the five transaction types.

Held-out evaluation (fraud is the positive class):

| Model | Accuracy | Fraud precision | Fraud recall | Fraud F1 |
|---|---|---|---|---|
| Decision tree | 95.217458% | 2.616586% | 99.513086% | 0.050991 |
| Always Not fraud | 99.870887% | 0.000000% | 0.000000% | 0.000000 |

Confusion matrices, `[[TN, FP], [FN, TP]]`:

```text
Decision tree:     [[1210030, 60851], [8, 1635]]
Always Not fraud:  [[1270881, 0], [1643, 0]]
```

The tree caught **1,635 of 1,643 frauds**, missed **8**, and raised **60,851 false alarms**. Although recall is high, only about **2.62% of its fraud alerts were correct**. Its low precision and F1 show that this fixed educational model would create a large review burden. The baseline has higher accuracy because fraud is rare, but it misses every fraud. These results were documented without changing parameters after seeing the test set.

The illustrative input above produced **Fraud** (`1`). This input has no known true label, so the prediction is not evidence that an actual transaction is fraudulent.

Export examples (the real files contain the complete results):

```json
{
  "decision_tree_accuracy": 0.9521745758822623,
  "illustrative": true,
  "prediction": 1,
  "prediction_label": "Fraud",
  "known_true_label": null
}
```

The final notebook cell printed `Run checks passed.` and displayed the saved JSON and comparison CSV. No execution-dependent value remains to be filled for this run.

<!-- EXECUTION_REPORT_END -->

## Saved files

- `outputs/evaluation_metrics.json`: dataset/schema information, inspection counts, feature exclusions, split settings, model parameters, package versions, tree/baseline metrics and confusion matrices.
- `outputs/example_prediction.json`: illustrative raw inputs, numeric prediction, readable label and `known_true_label: null`.
- `outputs/metrics_comparison.csv`: one row per model with accuracy, precision, recall and F1.

Outputs are displayed as well as saved. The model itself is not serialized; rerun the notebook to train it. `.gitignore` excludes dataset CSVs, virtual environments and credentials, while allowing the small metrics CSV.

## Libraries and functions

| Library / function | Purpose |
|---|---|
| `%pip install`, `requirements.txt` | Install dependencies into the chosen environment |
| KaggleHub `dataset_download()` | Download/cache the selected Kaggle dataset |
| `pathlib.Path`, `rglob()` | Find the CSV and save files without machine-specific paths |
| Pandas `read_csv()`, `head()`, `sample()`, `shape`, `dtypes` | Load and inspect tabular data |
| Pandas `isna().sum()`, `duplicated()`, `drop_duplicates()`, `value_counts()` | Inspect missingness, remove exact duplicates and count labels/types |
| Pandas `DataFrame`, `to_csv()` | Construct matching inputs and export metric tables |
| NumPy `asarray()`, `mean()`, `unique()` | Work with confusion matrices and check predictions |
| Plotly Express `pie()`, `imshow()` | Show required distributions and confusion heatmap |
| scikit-learn `train_test_split()` | Reproducible stratified 80/20 partition |
| `SimpleImputer`, `OneHotEncoder` | Fill missing predictors and encode transaction categories |
| `ColumnTransformer`, `Pipeline` | Apply column-specific processing fitted on training only |
| `DecisionTreeClassifier`, `fit()` | Learn classification rules from labeled training rows |
| Pipeline `score()`, `predict()` | Test accuracy and predicted labels with the same preprocessing |
| `DummyClassifier` | Establish an always-non-fraud baseline |
| `precision_score`, `recall_score`, `f1_score`, `confusion_matrix` | Evaluate rare-class detection and errors |
| Standard library `json`, `importlib.metadata.version()` | Save JSON and record dependency versions |
| IPython `display`, `Markdown` | Keep tables and computed explanations visible |
| JupyterLab, IPykernel | Interactive notebook interface and Python execution kernel |
| NBFormat, NBClient | Validate/read notebook structure and execute it from a fresh kernel |

## Limitations

This is synthetic data with simulation-specific patterns. A random split can put the same customers and neighboring time periods in both partitions, even when customer identifiers are excluded. The result can overestimate future performance; customer-aware and chronological validation would be needed for deployment. Fraud behavior can change, new transaction types can appear, and destination balances may be incomplete or unavailable. Ignoring unseen encoded categories avoids a crash but does not establish good performance on them.

The depth-limited tree can miss patterns. Class weighting can cause many false alarms, so fraud recall must be weighed against precision and operational cost. The example prediction has no ground truth and is not a calibrated risk score. This project is an assignment, not a production payment filter.

## GitHub feature branch and submission

This workspace was not a Git repository when first inspected. The project is now prepared on `feature/online-payments-fraud-detection`. The instructions below describe the submission workflow; pushing the branch does not open a pull request or submit its URL.

1. Open your assignment repository or your fork locally. If needed, clone it from GitHub using its actual URL. Copy this project's notebook, README, requirements, runner, `.gitignore` and small output files into that repository. Do not add the large dataset CSV. If the course expects a new repository, create one on GitHub first and follow its clone instructions.
2. From that repository folder, update the default branch and create a feature branch. These commands assume the default branch is `main`; use the repository's actual default branch if different. Commit or stash unrelated local changes before switching branches.

   ```powershell
   git switch main
   git pull --ff-only origin main
   git switch -c feature/online-payments-fraud-detection
   ```

3. Run the notebook, save visible outputs and inspect the files. Stage only the assignment deliverables, review the staged changes and commit:

   ```powershell
   git add prediction.ipynb README.md requirements.txt run_notebook.py .gitignore
   git add outputs/evaluation_metrics.json outputs/example_prediction.json outputs/metrics_comparison.csv
   git diff --cached --stat
   git diff --cached --check
   git commit -m "Complete online payments fraud prediction project"
   git push -u origin feature/online-payments-fraud-detection
   ```

4. On GitHub, choose **Compare & pull request**. Select the required assignment repository/default branch as the base and your feature branch as the compare branch. Describe the fraud project, chosen features, pipeline, baseline, actual metrics and successful execution checks. Open the pull request. For a fork, verify that the base points to the repository required by your course.
5. Copy the **pull request URL** and submit it through the assignment's specified TUMO Labs submission channel. Opening the pull request and submitting its URL are separate steps. Check the original assignment for the exact destination and review criteria.
6. **Peer review 1 (manual):** review one other participant's submission using the course rubric and leave the required feedback.
7. **Peer review 2 (manual):** review a second participant's submission and leave the required feedback. Completing the first review does not complete the second.

## Requirements checklist

Completed implementation:

- [x] Required package installation, including KaggleHub, Pandas, NumPy, Plotly and scikit-learn.
- [x] Portable KaggleHub download, CSV discovery and Pandas loading.
- [x] Samples, shape, types, missing values, duplicates and fraud/non-fraud counts.
- [x] Both required `plotly.express.pie()` charts and class imbalance explanation.
- [x] Explained features/target, missing-value handling, encoding, exclusions and after-transaction balance limitations.
- [x] Training-only preprocessing and reproducible stratified 80/20 split.
- [x] Decision tree with justified algorithm, training method and fixed parameters.
- [x] Explicit `model.score()`, precision, recall, F1, confusion matrix and always-non-fraud comparison; no test tuning.
- [x] Explicit `model.predict()` with matching inputs and an illustrative example label.
- [x] Local JSON/CSV exports with visible notebook outputs.
- [x] README with setup, testing, library/function purposes, model explanation, output descriptions, evaluation, example and limitations.
- [x] Feature branch, pull request, URL submission and two separate peer review instructions.

Execution and manual follow-up:

<!-- EXECUTION_CHECK_START -->
- [x] Full fresh-kernel execution completed; all 11 code cells and saved-result checks passed. Actual outputs are recorded above and in the notebook.
<!-- EXECUTION_CHECK_END -->
- [ ] Compare against the unavailable original assignment attachment.
- [ ] Commit on a feature branch, push and open the GitHub pull request.
- [ ] Submit the pull request URL to the course.
- [ ] Complete peer review 1.
- [ ] Complete peer review 2.
