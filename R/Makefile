RSCRIPT := Rscript

suppl: 20_supplement.pdf ziptex

ziptex:: 
	zip -9 -r supplement.zip 20_supplement.pdf 20_supplement.tex 20_supplement_files;

cleanziptex::
	rm supplement.zip

%.pdf:: %.Rmd; $(RSCRIPT) -e "require(rmarkdown); render('$<');"
