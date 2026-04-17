library(rmarkdown)
library(parallel)

base_r  <- "/Users/hoshi/Local Code/FINLYTS-G2-Case2/R"
out_dir <- "/Users/hoshi/Local Code/FINLYTS-G2-Case2/paper/appendices"

# Register a parallel backend so tune_grid() uses all physical cores.
# On M4 this gives ~6-8x speedup on the CV loop (13 folds, 10 cores).
n_cores <- max(1L, detectCores(logical = FALSE) - 1L)  # leave 1 core free
cat("Registering", n_cores, "parallel workers...\n")
if (requireNamespace("doParallel", quietly = TRUE)) {
  cl <- parallel::makePSOCKcluster(n_cores)
  doParallel::registerDoParallel(cl)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  cat("doParallel registered.\n")
} else {
  cat("doParallel not found — running single-threaded.\n")
}

scripts <- list(
  list(rmd="01_data_pull.Rmd",                  pdf="01_data_pull.pdf",           engine="pdflatex"),
  list(rmd="02_feature_engineering_render.Rmd", pdf="02_feature_engineering.pdf", engine="xelatex"),
  list(rmd="03_ols_baseline_render.Rmd",        pdf="03_ols_baseline.pdf",        engine="pdflatex"),
  list(rmd="04_models_rf_xgboost.Rmd",          pdf="04_models_rf_xgboost.pdf",   engine="pdflatex"),
  list(rmd="05_models_svm_ann.Rmd",             pdf="05_models_svm_ann.pdf",      engine="pdflatex"),
  list(rmd="06_evaluation.Rmd",                 pdf="06_evaluation.pdf",          engine="pdflatex")
)

for (s in scripts) {
  rmd_path <- file.path(base_r, s$rmd)
  pdf_path <- file.path(out_dir, s$pdf)
  cat("\n=== Knitting", s$rmd, "===\n")
  tryCatch({
    render(
      input         = rmd_path,
      output_format = pdf_document(
        latex_engine    = s$engine,
        keep_tex        = FALSE,
        toc             = TRUE,
        number_sections = TRUE
      ),
      output_file = pdf_path,
      envir       = new.env(parent = globalenv()),
      quiet       = FALSE
    )
    cat("  -> Written:", pdf_path, "\n")
  }, error = function(e) {
    cat("  !! ERROR in", s$rmd, ":", conditionMessage(e), "\n")
  })
}
cat("\nAll done.\n")
