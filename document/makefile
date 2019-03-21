help:
	@echo "Available makes:"
	@echo "all abs eabs gp gpe gp2c clean cleanall"

	make gp
#	make gp2c


# Geophysics example
gp:
	pdflatex cc.ltx
	bibtex   cc
	pdflatex cc.ltx
	pdflatex cc.ltx

clean:
	rm *.aux *.bbl *.blg *.log 

#cleanall:
#	rm *.aux *.bbl *.blg *.log 

