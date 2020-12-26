# Description of data balance in the four strategies

## Histograms - ratio distributions

Trends:

* `resampling_after_clustering` keeps balanced proteins.
* `resampling_before_clustering` and `semi_resampling` lead to a more balanced train test, but not so much for the test set.
* `no_resampling` keeps similar data imbalance in training and test.
*  Tests sets with imbalance tend to create extremes in imbalance (all actives or all inactives), probably due to the combination of the clustering and the lower sample sizes in test.

## Scatterplots - comparing training and test

Trends:

* `no_resampling` keeps a positive relation between both, i.e. proteins tend to keep their balance in train and test.
* `resampling_after_clustering` keeps balanced proteins.
* `resampling_before_clustering` shows an inverse relationship instead -- this is expected since it starts from globally balanced proteins, and after the clustering, an imbalance in one direction in the training set entails an inverse imbalance in the test set.
* `semi_resampling` leads to independent train and test balances, expected since the train set is resampled, breaking any correlation with the test set balance.

# Linear models on predicted proportions

## Histograms - predicted ratio distributions

Trends:

* `no_resampling` is noticeably biased to predict everything as positives.
* `resampling_after_clustering` keeps a wide and symmetric distribution of predicted actives.
* `resampling_before_clustering` and `semi_resampling` alleviate the imbalance in the predictions, but still retain a spike of proteins where all the compounds are predicted as positives.

## Scatterplots - predicted proportions against training proportions

Trends:

* `no_resampling`: positive trend between the training and the predicted ratio, but since the training and the test ratio also correlate, the latter could be the one driving the predicted ratio of positives.
* `resampling_after_clustering` has a contant training ratio, meaning that it cannot explain the predicted ratio.
* `resampling_before_clustering` shows instead a negative relation between the training and the predicted ratio, but since the former and the test ratio also anticorrelate, the simplest explanation is that the test ratio drives the predicted test ratio.
* `semi_resampling` shows independence between the predicted ratio and the training ratio.

# Linear models on baseline performance

* Describe baseline metrics
* Classify them as sensitive (ACC, F1, BACC)/insensitive (AUROC, MCC) by data balance
* Direct comparison of imbalance-sensitive metrics can be misleading (proof? boxplots of absolute vs boxplot of diffs)

## Imbalance-insensitive

* Models on performance values directly
* Largest drive of performance: augmenting test set: artificial boost?
* Without augmenting test set, resampling is slightly preferable (AUROC not improving, MCC improving)
* Effect of data availability? Of train/test set ratio?

## Imbalance-sensitive

* Models on perf-baselineperf
* Largest drive of performance: augmenting test set
* Without augmenting test set, resampling is slightly preferable
* More data availability (number of interactions) increases performance




# Notes

## Latex tables

https://stackoverflow.com/questions/55308088/floating-toc-doesnt-work-in-interactive-rmarkdown
https://stackoverflow.com/questions/16507191/automatically-adjust-latex-table-width-to-fit-pdf-using-knitr-and-rstudio
https://stackoverflow.com/questions/54082814/adding-label-in-kable-kableextra-latex-output



# Story



```{r, results='asis'}
table(df.clean.nobaseline$strategy, df.clean.nobaseline$fold.label) %>%
  as.data.frame.matrix() %>%
  tibble::rownames_to_column("Strategy") %>%
  xtable::xtable(
    caption = "Number of proteins for which performance metrics were computed.", 
    label = "tab:nprot-fold-strat") %>% 
  print(include.rownames = FALSE, comment = FALSE)
```


\begin{table}
\resizebox{\textwidth}{!} {
```{r, results='asis'}
table(df.clean.nobaseline$strategy, df.clean.nobaseline$fold.label) %>%
  as.data.frame.matrix() %>%
  tibble::rownames_to_column("Strategy") %>%
  xtable::xtable() %>% 
  print(include.rownames = FALSE, comment = FALSE)
```
}
\caption{Number of proteins for which performance metrics were computed.}
\label{tab:nprot-fold-strat}
\end{table}