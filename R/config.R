gg_theme <- ggplot2::theme_bw() +
    ggplot2::theme(text = element_text(size = 7.5))

gg_45 <- ggplot2::theme(
    axis.text.x = ggplot2::element_text(angle = 45, vjust = 1, hjust = 1))

v.metrics <- c("acc", "auroc", "f1", "balanced_acc", "mcc")

v.strategy <- c(
    "no_resampling", 
    "resampling_before_clustering", 
    "semi_resampling", 
    "resampling_after_clustering")
col.strategy <- setNames(
    c("lightpink1", 
      "powderblue", 
      "mediumpurple1", 
      "aquamarine2"), 
    v.strategy)

star.cutoffs <- c(.05, .001, .000001)

dir.interim <- "data-interim"
if (!dir.exists(dir.interim)) dir.create(dir.interim)
file.cleanperf <- paste0(dir.interim, "/ratios_df_clean.csv")
file.cleanperf.gpcr <- paste0(dir.interim, "/ratios_df_clean_gpcr.csv")
file.cleanperf.nr <- paste0(dir.interim, "/ratios_df_clean_nr.csv")
file.cleanperf.reglin <- paste0(dir.interim, "/ratios_df_clean_reglin.csv")
