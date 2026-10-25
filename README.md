# CART Student Risk Lab

A local academic prototype that trains a real Gini CART classifier on the Open University Learning Analytics Dataset (OULAD), predicts whether an eligible course enrollment will eventually be marked `Withdrawn`, and explains each root-to-leaf split. It is built as a fresh local app because the public CART Decision Lab source repository named in the supplied report was unavailable.

## Scope and data

- Source: UCI OULAD, DOI [10.24432/C5KK69](https://doi.org/10.24432/C5KK69), CC BY 4.0. Dataset description: [Kuzilek et al., Scientific Data, DOI 10.1038/sdata.2017.171](https://doi.org/10.1038/sdata.2017.171).
- Target: `final_result == Withdrawn` versus `Pass`, `Fail`, or `Distinction`. Fail is not merged with Withdrawn.
- Cutoff: end of day 28 from course start. Eligible registrations have `date_registration <= 28` and no `date_unregistration <= 28`; records with missing registration dates are excluded. VLE features aggregate days 0–28 inclusive.
- `final_result`, student identifiers, registration dates, and unregistration dates are not model features. Unregistration and registration dates only define cutoff eligibility. No demographics are used as predictors.
- A GroupShuffleSplit by anonymized student ID holds out 20% for testing. Five fold StratifiedGroupKFold on training students selects the pruning configuration. The predeclared model uses Gini, class weights balanced, maximum depth 8, then the simplest candidate within one standard error of best training CV F1 for Withdrawn.
- The prediction score is the class fraction in the tree leaf. It is not a calibrated probability.

## Requirements

Python 3.12 or a compatible Python with pandas, NumPy, scikit-learn, and joblib. The Codex bundled Python environment used to create this demo has these packages already available. Exact versions are recorded in `artifacts/metrics.json` and can be locked in `requirements.txt`.

## Run

1. Download the official UCI OULAD archive and save it as `data/raw/oulad.zip`. Raw dataset files are excluded from this Git repository.
2. Extract it to `data/raw/oulad/`, so `studentInfo.csv`, `studentRegistration.csv`, and `studentVle.csv` are directly inside that folder.
3. Train and regenerate the model and metrics:

   ```powershell
   python ml/train.py
   ```

4. Start the app:

   ```powershell
   python app/server.py
   ```

5. Open `http://127.0.0.1:8765`.

## Deploy on a Python web host

GitHub stores the source but does not run this Python app. On a Python web host, use build command `pip install -r requirements.txt`, start command `python app/server.py`, and set `HOST=0.0.0.0`. The app reads the host-provided `PORT` variable; locally it defaults to `127.0.0.1:8765`.

The bundled `artifacts/` already contains a trained model, its schema, and the evaluation from the completed training run. The app serves those artifacts locally; it does not upload student data or call an external inference service.

## Verification

```powershell
python -m unittest discover -s tests -v
node --check app/static/app.js
```

The browser smoke check should confirm the four navigation pages render, one example prediction returns a CART decision path, and invalid input is rejected. The static frontend has no separate bundler/build step. A local preview does not confirm Vercel or other production deployment.

## Important limitations

OULAD is anonymized historical data from the Open University, not current local student data. The test metrics are modest and reflect an intentionally honest group holdout; at the chosen operating point the model favors recall over precision and generates many false alerts. This prototype must not be used to label, discipline, rank, or automatically make decisions about individual students. Any real intervention requires current data, consent and governance review, calibration and fairness evaluation, external validation, and human judgment.
