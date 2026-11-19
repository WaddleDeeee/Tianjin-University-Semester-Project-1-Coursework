# BCI EEG Feature Extraction and Classification

Python programs for four connected EEG assignments using BCI Competition IV 2a data. The `archive/` folder contains the EEG recordings used by the programs.

- `main.py`: preprocess EEG trials and extract time, frequency, and CSP features. Results are written to `output/`.
- `task2.py`: feature selection and PCA; writes tables and figures to `output/task2/`.
- `task3.py`: K-means and DBSCAN clustering on task 2 PCA features; writes results to `output/task3/`.
- `task4.py`: train a decision tree and SVM on session T, evaluate on session E; writes results to `output/task4/`.

`作业2/` contains anonymized copies of the four assignment deliverables. Run the scripts in order with Python 3 and the packages used in the source files (NumPy, SciPy, pandas, MNE, scikit-learn, matplotlib, openpyxl, joblib; Word report generation also uses python-docx).

Example: `python main.py`, then `python task2.py`, `python task3.py`, and `python task4.py`. See each script's `--help` for input and output options.
