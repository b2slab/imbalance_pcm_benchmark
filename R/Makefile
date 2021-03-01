# R version to use in the analysis (defaults to system version)
RSCRIPT := Rscript

## all the supplement analysis
all: supplement.zip

## steps
## 
# 0: data summary
10_overview_allfamilies_foldavg.tex: 10_overview_allfamilies.Rmd
	$(RSCRIPT) -e "require(rmarkdown); render('$<');";

# 1: data cleaning
data-interim/ratios_df_clean.csv: 12_performance_explorer.Rmd
	$(RSCRIPT) -e "require(rmarkdown); render('$<');";
data-interim/ratios_df_clean_gpcr.csv: 13_performance_explorer_gpcr.Rmd
	$(RSCRIPT) -e "require(rmarkdown); render('$<');";
data-interim/ratios_df_clean_nr.csv: 14_performance_explorer_nr.Rmd
	$(RSCRIPT) -e "require(rmarkdown); render('$<');";
data-interim/ratios_df_clean_pr.csv: 15_performance_explorer_pr.Rmd
	$(RSCRIPT) -e "require(rmarkdown); render('$<');";

# 2: data analysis
20_supplement_kinases.pdf: 20_supplement_kinases.Rmd data-interim/ratios_df_clean.csv
21_supplement_gpcr.pdf: 21_supplement_gpcr.Rmd data-interim/ratios_df_clean_gpcr.csv
22_supplement_nr.pdf: 22_supplement_nr.Rmd data-interim/ratios_df_clean_nr.csv
23_supplement_pr.pdf: 23_supplement_pr.Rmd data-interim/ratios_df_clean_pr.csv

# 3: zip files
supplement.zip: 10_overview_allfamilies_foldavg.tex 20_supplement_kinases.pdf 21_supplement_gpcr.pdf 22_supplement_nr.pdf
	rm -f supplement.zip;
	zip -9 -r supplement.zip \
		10_overview_allfamilies_foldavg.tex \
		20_supplement_kinases.pdf 20_supplement_kinases.tex 20_supplement_kinases_files \
		21_supplement_gpcr.pdf 21_supplement_gpcr.tex 21_supplement_gpcr_files \
		22_supplement_nr.pdf 22_supplement_nr.tex 22_supplement_nr_files \
		23_supplement_pr.pdf 23_supplement_pr.tex 23_supplement_pr_files;

## general recipes
%.pdf:: %.Rmd; $(RSCRIPT) -e "require(rmarkdown); render('$<');";
%.html:: %.Rmd; $(RSCRIPT) -e "require(rmarkdown); render('$<');";
