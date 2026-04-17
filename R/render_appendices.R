library(rmarkdown)
base_r   <- "/Users/hoshi/Local Code/FINLYTS-G2-Case2/R"
out_dir  <- "/Users/hoshi/Local Code/FINLYTS-G2-Case2/paper/appendices"

scripts <- list(
  list(rmd="01_data_pull.Rmd", pdf="01_data_pull.pdf"),
  list(rmd="02_feature_engineering_render.Rmd", pdf="02_feature_engineering.pdf"),
  list(rmd="03_ols_baseline_render.Rmd", pdf="03_ols_baseline.pdf"),
  list(rmd="04_models_rf_xgboost_render.Rmd", pdf="04_models_rf_xgboost.pdf"),
  list(rmd="05_models_svm_ann_render.Rmd", pdf="05_models_svm_ann.pdf"),
  list(rmd="06_evaluation.Rmd", pdf="06_evaluation.pdf")
)

for (s in scripts) {
  rmd_path <- file.path(base_r, s$rmd)
  pdf_path <- file.path(out_dir, s$pdf)
  cat("\n=== Knitting", s$rmd, "===\n")
  tryCatch({
    render(
      input          = rmd_path,
      output_format  = pdf_document(
        latex_engine = "pdflatex",
        keep_tex     = FALSE,
        toc          = TRUE,
        number_sections = TRUE
      ),
      output_file    = pdf_path,
      envir          = new.env(parent = globalenv()),
      quiet          = FALSE
    )
    cat("  -> Written:", pdf_path, "\n")
  }, error = function(e) {
    cat("  !! ERROR in", s$rmd, ":", conditionMessage(e), "\n")
  })
}
cat("\nAll done.\n")
