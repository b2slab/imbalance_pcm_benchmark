RSCRIPT := Rscript

suppl: 20_supplement_kinases.pdf 21_supplement_gpcr.pdf ziptex

ziptex:: 
	zip -9 -r supplement.zip \
		20_supplement_kinases.pdf 20_supplement_kinases.tex 20_supplement_kinases_files \
		21_supplement_gpcr.pdf 21_supplement_gpcr.tex 21_supplement_gpcr_files;

cleanziptex::
	rm supplement.zip

%.pdf:: %.Rmd; $(RSCRIPT) -e "require(rmarkdown); render('$<');"
