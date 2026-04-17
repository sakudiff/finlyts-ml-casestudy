#!/usr/bin/env python3
"""
Patch scripts 04 and 05 to load pre-saved .rds models instead of re-training,
then write an R render script that knits all 6 Rmds to PDF.
"""
import re, os, textwrap

BASE = os.path.dirname(os.path.abspath(__file__))

# ── RF LOADER CHUNK (injected after rf_models eval=FALSE) ──────────────────
RF_LOAD = r'''
```{r rf_load, echo=FALSE, message=TRUE}
# ── Pre-trained RF models loaded from .rds ──
rf_metrics_list <- vector("list", length(TARGET_COLS))
for (h in seq_along(TARGET_COLS)) {
  target_col    <- TARGET_COLS[h]
  horizon_label <- HORIZON_LABS[h]
  train_h <- train_df %>% select(all_of(c(target_col, PREDICTOR_COLS)))
  test_h  <- test_df  %>% select(all_of(c(target_col, PREDICTOR_COLS)))
  final_rf_fit   <- readRDS(here("output", "models", paste0("rf_", horizon_label, ".rds")))
  rf_preds       <- predict(final_rf_fit, new_data = test_h)$.pred
  rf_train_preds <- predict(final_rf_fit, new_data = train_h)$.pred
  naive_preds    <- test_df$lag1
  rf_metrics_list[[h]] <- compute_metrics(
    test_actual  = test_h[[target_col]], test_pred    = rf_preds,
    train_actual = train_h[[target_col]], train_pred  = rf_train_preds,
    naive_pred   = naive_preds, model = "Random Forest", horizon = horizon_label
  )
  cat("RF", toupper(horizon_label),
      "| RMSE:", round(rf_metrics_list[[h]]$RMSE, 6),
      "| DA:",   round(rf_metrics_list[[h]]$DA,   4), "\n")
  print(
    final_rf_fit %>%
      extract_fit_parsnip() %>%
      vip(num_features = 15, aesthetics = list(fill = "#141F52", alpha = 0.9)) +
      labs(title    = "Random Forest \u2014 Feature Importance (Top 15)",
           subtitle = paste0(toupper(horizon_label), " Horizon | Test RMSE: ",
                             round(rf_metrics_list[[h]]$RMSE, 6)),
           caption  = "Source: Custom ML Pipeline. 20% holdout.",
           x = "Impurity-Based Importance", y = NULL) +
      theme_quant()
  )
  print(
    ggplot(tibble(Actual = test_h[[target_col]], Predicted = rf_preds),
           aes(x = Actual, y = Predicted)) +
      geom_point(color = "#141F52", alpha = 0.6) +
      geom_abline(intercept = 0, slope = 1, color = "#E3120B",
                  linetype = "dashed", linewidth = 1) +
      labs(title    = "Random Forest \u2014 Predicted vs Actual Returns",
           subtitle = paste0(toupper(horizon_label), " Horizon | Test RMSE: ",
                             round(rf_metrics_list[[h]]$RMSE, 6)),
           caption  = "Source: Custom ML Pipeline. 20% holdout.",
           x = "Actual Log Return", y = "Predicted Log Return") +
      theme_quant()
  )
}
rf_metrics <- bind_rows(rf_metrics_list)
```
'''

# ── XGB LOADER CHUNK ───────────────────────────────────────────────────────
XGB_LOAD = r'''
```{r xgb_load, echo=FALSE, message=TRUE}
# ── Pre-trained XGBoost models loaded from .rds ──
xgb_metrics_list <- vector("list", length(TARGET_COLS))
for (h in seq_along(TARGET_COLS)) {
  target_col    <- TARGET_COLS[h]
  horizon_label <- HORIZON_LABS[h]
  train_h <- train_df %>% select(all_of(c(target_col, PREDICTOR_COLS)))
  test_h  <- test_df  %>% select(all_of(c(target_col, PREDICTOR_COLS)))
  final_xgb_fit   <- readRDS(here("output", "models", paste0("xgboost_", horizon_label, ".rds")))
  xgb_preds       <- predict(final_xgb_fit, new_data = test_h)$.pred
  xgb_train_preds <- predict(final_xgb_fit, new_data = train_h)$.pred
  naive_preds     <- test_df$lag1
  xgb_metrics_list[[h]] <- compute_metrics(
    test_actual  = test_h[[target_col]], test_pred    = xgb_preds,
    train_actual = train_h[[target_col]], train_pred  = xgb_train_preds,
    naive_pred   = naive_preds, model = "XGBoost", horizon = horizon_label
  )
  cat("XGBoost", toupper(horizon_label),
      "| RMSE:", round(xgb_metrics_list[[h]]$RMSE, 6),
      "| DA:",   round(xgb_metrics_list[[h]]$DA,   4), "\n")
  print(
    final_xgb_fit %>%
      extract_fit_parsnip() %>%
      vip(num_features = 15, aesthetics = list(fill = "#141F52", alpha = 0.9)) +
      labs(title    = "XGBoost \u2014 Feature Importance (Top 15)",
           subtitle = paste0(toupper(horizon_label), " Horizon | Test RMSE: ",
                             round(xgb_metrics_list[[h]]$RMSE, 6)),
           caption  = "Source: Custom ML Pipeline. 20% holdout.",
           x = "Importance (Gain)", y = NULL) +
      theme_quant()
  )
  print(
    ggplot(tibble(Actual = test_h[[target_col]], Predicted = xgb_preds),
           aes(x = Actual, y = Predicted)) +
      geom_point(color = "#141F52", alpha = 0.6) +
      geom_abline(intercept = 0, slope = 1, color = "#E3120B",
                  linetype = "dashed", linewidth = 1) +
      labs(title    = "XGBoost \u2014 Predicted vs Actual Returns",
           subtitle = paste0(toupper(horizon_label), " Horizon | Test RMSE: ",
                             round(xgb_metrics_list[[h]]$RMSE, 6)),
           caption  = "Source: Custom ML Pipeline. 20% holdout.",
           x = "Actual Log Return", y = "Predicted Log Return") +
      theme_quant()
  )
}
xgb_metrics <- bind_rows(xgb_metrics_list)
```
'''

# ── SVM LOADER CHUNK ───────────────────────────────────────────────────────
SVM_LOAD = r'''
```{r svm_load, echo=FALSE, message=TRUE}
# ── Pre-trained SVM models loaded from .rds ──
svm_metrics_list <- vector("list", length(TARGET_COLS))
for (h in seq_along(TARGET_COLS)) {
  target_col    <- TARGET_COLS[h]
  horizon_label <- HORIZON_LABS[h]
  train_h <- train_df %>% select(all_of(c(target_col, PREDICTOR_COLS)))
  test_h  <- test_df  %>% select(all_of(c(target_col, PREDICTOR_COLS)))
  final_svm_fit   <- readRDS(here("output", "models", paste0("svm_", horizon_label, ".rds")))
  svm_preds       <- predict(final_svm_fit, new_data = test_h)$.pred
  svm_train_preds <- predict(final_svm_fit, new_data = train_h)$.pred
  naive_preds     <- test_df$lag1
  svm_metrics_list[[h]] <- compute_metrics(
    test_actual  = test_h[[target_col]], test_pred    = svm_preds,
    train_actual = train_h[[target_col]], train_pred  = svm_train_preds,
    naive_pred   = naive_preds, model = "SVM", horizon = horizon_label
  )
  cat("SVM", toupper(horizon_label),
      "| RMSE:", round(svm_metrics_list[[h]]$RMSE, 6),
      "| DA:",   round(svm_metrics_list[[h]]$DA,   4), "\n")
  print(
    ggplot(tibble(Actual = test_h[[target_col]], Predicted = svm_preds),
           aes(x = Actual, y = Predicted)) +
      geom_point(color = "#141F52", alpha = 0.6) +
      geom_abline(intercept = 0, slope = 1, color = "#E3120B",
                  linetype = "dashed", linewidth = 1) +
      labs(title    = "SVM \u2014 Predicted vs Actual Returns",
           subtitle = paste0(toupper(horizon_label), " Horizon | Test RMSE: ",
                             round(svm_metrics_list[[h]]$RMSE, 6)),
           caption  = "Source: Custom ML Pipeline. 20% holdout.",
           x = "Actual Log Return", y = "Predicted Log Return") +
      theme_quant()
  )
}
svm_metrics <- bind_rows(svm_metrics_list)
```
'''

# ── ANN LOADER CHUNK ───────────────────────────────────────────────────────
ANN_LOAD = r'''
```{r ann_load, echo=FALSE, message=TRUE}
# ── Pre-trained ANN models loaded from .rds ──
ann_metrics_list <- vector("list", length(TARGET_COLS))
for (h in seq_along(TARGET_COLS)) {
  target_col    <- TARGET_COLS[h]
  horizon_label <- HORIZON_LABS[h]
  train_h <- train_df %>% select(all_of(c(target_col, PREDICTOR_COLS)))
  test_h  <- test_df  %>% select(all_of(c(target_col, PREDICTOR_COLS)))
  final_ann_fit   <- readRDS(here("output", "models", paste0("ann_", horizon_label, ".rds")))
  ann_preds       <- predict(final_ann_fit, new_data = test_h)$.pred
  ann_train_preds <- predict(final_ann_fit, new_data = train_h)$.pred
  naive_preds     <- test_df$lag1
  ann_metrics_list[[h]] <- compute_metrics(
    test_actual  = test_h[[target_col]], test_pred    = ann_preds,
    train_actual = train_h[[target_col]], train_pred  = ann_train_preds,
    naive_pred   = naive_preds, model = "ANN", horizon = horizon_label
  )
  cat("ANN", toupper(horizon_label),
      "| RMSE:", round(ann_metrics_list[[h]]$RMSE, 6),
      "| DA:",   round(ann_metrics_list[[h]]$DA,   4), "\n")
  print(
    ggplot(tibble(Actual = test_h[[target_col]], Predicted = ann_preds),
           aes(x = Actual, y = Predicted)) +
      geom_point(color = "#141F52", alpha = 0.6) +
      geom_abline(intercept = 0, slope = 1, color = "#E3120B",
                  linetype = "dashed", linewidth = 1) +
      labs(title    = "ANN \u2014 Predicted vs Actual Returns",
           subtitle = paste0(toupper(horizon_label), " Horizon | Test RMSE: ",
                             round(ann_metrics_list[[h]]$RMSE, 6)),
           caption  = "Source: Custom ML Pipeline. 20% holdout.",
           x = "Actual Log Return", y = "Predicted Log Return") +
      theme_quant()
  )
}
ann_metrics <- bind_rows(ann_metrics_list)
```
'''

def patch_rmd(src_path, dest_path, patches):
    """
    patches: list of (chunk_label, loader_code) tuples.
    For each chunk_label, marks it eval=FALSE and inserts loader_code after its closing ```.
    """
    with open(src_path, encoding="utf-8") as f:
        text = f.read()

    for label, loader in patches:
        # Match the opening fence of the named chunk and add eval=FALSE
        # Pattern: ```{r label} or ```{r label, ...}
        open_pat = re.compile(
            r'(```\{r ' + re.escape(label) + r')([ ,}])',
            re.MULTILINE
        )
        def add_eval_false(m):
            prefix = m.group(1)
            suffix = m.group(2)
            if suffix == '}':
                return prefix + ', eval=FALSE}'
            else:
                return prefix + ', eval=FALSE' + suffix
        text = open_pat.sub(add_eval_false, text, count=1)

        # Find the closing ``` of that chunk and insert loader after it
        # We locate the chunk start, then find next standalone ```
        chunk_start = re.search(r'```\{r ' + re.escape(label) + r'[, }]', text)
        if chunk_start:
            # Find next ``` on its own line after the chunk start
            close_match = re.search(r'^```\s*$', text[chunk_start.end():], re.MULTILINE)
            if close_match:
                insert_pos = chunk_start.end() + close_match.end()
                text = text[:insert_pos] + '\n' + loader + text[insert_pos:]

    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Patched → {os.path.basename(dest_path)}")


# ── Patch script 02: replace ✓ with Y (pdflatex can't handle U+2713) ──────
def text_fix(src_path, dest_path, replacements):
    with open(src_path, encoding="utf-8") as f:
        text = f.read()
    for old, new in replacements:
        text = text.replace(old, new)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Text-fixed → {os.path.basename(dest_path)}")

text_fix(
    src_path  = os.path.join(BASE, "02_feature_engineering.Rmd"),
    dest_path = os.path.join(BASE, "02_feature_engineering_render.Rmd"),
    replacements = [
        ("✓", "Yes"),
    ]
)

# ── Patch script 03: escape \linewidth in Markdown prose ──────────────────
text_fix(
    src_path  = os.path.join(BASE, "03_ols_baseline.Rmd"),
    dest_path = os.path.join(BASE, "03_ols_baseline_render.Rmd"),
    replacements = [
        ("tabular* with \\linewidth for perfect fit.",
         "tabular* with linewidth for perfect fit."),
    ]
)

# ── Patch script 04 ────────────────────────────────────────────────────────
patch_rmd(
    src_path  = os.path.join(BASE, "04_models_rf_xgboost.Rmd"),
    dest_path = os.path.join(BASE, "04_models_rf_xgboost_render.Rmd"),
    patches   = [("rf_models", RF_LOAD), ("xgb_models", XGB_LOAD)]
)

# ── Patch script 05 ────────────────────────────────────────────────────────
patch_rmd(
    src_path  = os.path.join(BASE, "05_models_svm_ann.Rmd"),
    dest_path = os.path.join(BASE, "05_models_svm_ann_render.Rmd"),
    patches   = [("svm_models", SVM_LOAD), ("ann_models", ANN_LOAD)]
)

# ── Write R render script ──────────────────────────────────────────────────
project_root = os.path.dirname(BASE)
out_dir = os.path.join(project_root, "paper", "appendices")
os.makedirs(out_dir, exist_ok=True)

scripts = [
    ("01_data_pull.Rmd",                   "01_data_pull.Rmd"),
    ("02_feature_engineering_render.Rmd",  "02_feature_engineering.Rmd"),
    ("03_ols_baseline_render.Rmd",         "03_ols_baseline.Rmd"),
    ("04_models_rf_xgboost_render.Rmd",    "04_models_rf_xgboost.Rmd"),
    ("05_models_svm_ann_render.Rmd",       "05_models_svm_ann.Rmd"),
    ("06_evaluation.Rmd",                  "06_evaluation.Rmd"),
]

r_lines = [
    'library(rmarkdown)',
    f'base_r   <- "{BASE}"',
    f'out_dir  <- "{out_dir}"',
    '',
    'scripts <- list(',
]
for rmd_file, out_stem in scripts:
    out_name = os.path.splitext(out_stem)[0] + ".pdf"
    r_lines.append(f'  list(rmd="{rmd_file}", pdf="{out_name}"),')
r_lines[-1] = r_lines[-1].rstrip(",")  # remove trailing comma on last item
r_lines += [
    ')',
    '',
    'for (s in scripts) {',
    '  rmd_path <- file.path(base_r, s$rmd)',
    '  pdf_path <- file.path(out_dir, s$pdf)',
    '  cat("\\n=== Knitting", s$rmd, "===\\n")',
    '  tryCatch({',
    '    render(',
    '      input          = rmd_path,',
    '      output_format  = pdf_document(',
    '        latex_engine = "pdflatex",',
    '        keep_tex     = FALSE,',
    '        toc          = TRUE,',
    '        number_sections = TRUE',
    '      ),',
    '      output_file    = pdf_path,',
    '      envir          = new.env(parent = globalenv()),',
    '      quiet          = FALSE',
    '    )',
    '    cat("  -> Written:", pdf_path, "\\n")',
    '  }, error = function(e) {',
    '    cat("  !! ERROR in", s$rmd, ":", conditionMessage(e), "\\n")',
    '  })',
    '}',
    'cat("\\nAll done.\\n")',
]

render_r_path = os.path.join(BASE, "render_appendices.R")
with open(render_r_path, "w", encoding="utf-8") as f:
    f.write("\n".join(r_lines) + "\n")
print(f"  R render script → {render_r_path}")
print("Done. Run: Rscript R/render_appendices.R")
