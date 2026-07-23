# Upload instructions

These files are ready to place at the root of the GitHub repository:

https://github.com/kensuenck/Predicting-Secondary-School-Academic-Risk-Using-Machine-Learning

## Using the GitHub website

1. Extract the ZIP package.
2. Open the repository on GitHub.
3. Choose **Add file → Upload files**.
4. Upload the contents of the extracted root folder, preserving the folder structure.
5. Use a commit message such as:

   `Release dissertation reproducibility package v1.0.0`

6. After checking the files, create a release named `v1.0-dissertation` and attach the ZIP package if desired.

## Using Git

```bash
git clone https://github.com/kensuenck/Predicting-Secondary-School-Academic-Risk-Using-Machine-Learning.git
cd Predicting-Secondary-School-Academic-Risk-Using-Machine-Learning
```

Copy the package contents into the cloned folder, then run:

```bash
git add .
git commit -m "Release dissertation reproducibility package v1.0.0"
git push origin main
git tag -a v1.0-dissertation -m "Dissertation submission release"
git push origin v1.0-dissertation
```

## Final checks on GitHub

- Confirm that `README.md` renders correctly.
- Confirm that `CITATION.cff` creates a **Cite this repository** panel.
- Confirm that the GitHub Actions reproduction workflow starts or can be run manually.
- Confirm that the dissertation PDF opens.
- Confirm that no emails, assessment reports, passwords, tokens or private files were uploaded.
