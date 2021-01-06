RSCRIPT := Rscript

suppl: 20_supplement_kinases.pdf ziptex

ziptex:: 
	zip -9 -r supplement.zip 20_supplement_kinases.pdf \
		20_supplement_kinases.tex 20_supplement_kinases_files;

cleanziptex::
	rm supplement.zip

%.pdf:: %.Rmd; $(RSCRIPT) -e "require(rmarkdown); render('$<');"
