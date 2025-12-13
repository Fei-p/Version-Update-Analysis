## Version-Update-Analysis
This repo contains the code and data for our analysis project on the S14 (v1.3.3) version update of a global weather app.
We study how the update changed user behavior, monetization, and retention across key markets.

#### Repository structure :
1. final-version-data-augmented.csv: Main user-level dataset used in all analyses (daily observations with version, country, usage, ad metrics, and retention flags).
2. Fin_script_221.Rmd: Main R Markdown script. This Runs the full analysis used in the report (EDA, PCA, regression, trees, GBM, logistic models), and produces the tables and figures referenced in the paper (Parts 1–3: behavior, monetization, retention).
3. bootstrap.py : Implements a reusable run_bootstrap_analysis() function for comparing v1.2.9 vs v1.3.3 on a given metric (e.g., session duration, ad impressions), Draws bootstrap samples, computes the distribution of mean differences, and plots a histogram with 95% CIs.
4. lasso.py: Builds LASSO models to identify drivers of ad revenue separately for each country (e.g., page_home, session_open, ad_density, rewarded_ratio, etc.), Standardizes features, uses cross-validated LASSO, and prints non-zero coefficients as the main drivers.
5. pca.py: Runs PCA on behavioral features (session frequency, depth, ad density, page views, CTR, etc.) for USA vs India, produces a 2D PC scatterplot and loading table to visualize cross-market behavioral patterns.
6. decision tree.py: Fits a shallow decision tree (regressor) relating revenue to core behavioral features (e.g., page_home, ad_density, avg_duration_per_session) for a single country, exports tree rules and a visualization that highlights key behavioral thresholds.

#### Reproducing the R analysis
- Open the project folder in RStudio.
- Make sure your working directory is the repo root (where the .Rmd and .csv files live).
- Install required R packages (if you don’t already have them):
  ```
  install.packages(c(
  "tidyverse",
  "ggplot2",
  "dplyr",
  "purrr",
  "rlang",
  "glmnet",
  "broom",
  "scales",
  "rpart",
  "rpart.plot",
  "gbm",
  "pROC",
  "DT",
  "knitr",
  "tibble",
  "kableExtra"
  ))
  ```
  
- Open Fin_script_221.Rmd
- Knit to HTML (or run chunks top-to-bottom) to reproduce all tables and figures used in the report.

##### Analysis overview in R
At a high level, the project proceeds in three stages:
1. User behaviour (Part 1)
   - Compare behaviour between v1.2.9 and v1.3.3 using regression and bootstrap.
   - Use PCA to define engagement components summarising session frequency, depth, and ad exposure.
2. Monetization (Part 2)
   - Model log(ad_revenue + 1) using OLS, ridge, and lasso with version, country, calendar variables and PCA components.
   - Reframe monetization as a classification problem (high-value = top 25% revenue) and fit tree and GBM models.
3. Retention (Part 3)
   - Model 7-day retention using logistic regression, classification trees, and GBM.
   - Use the same engagement components to link behaviour, monetization, and retention.
Fin_script_221.Rmd is the single entry point to reproduce all figures and tables used in the written report.

#### Reproducing Python Script
The Python files are supporting analyses that mirror or extend parts of the R work.
All scripts expect the dataset in the repo root:
```
df = pd.read_csv("final-version-data-augmented.csv")
```
Example usage from the command line:
```
python bootstrap.py
python lasso.py
python "decision tree.py"
python pca.py
```
  
