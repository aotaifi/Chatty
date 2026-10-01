PYTHON ?= python3

figures:
	$(PYTHON) scripts/make_figures.py

pdf: figures
	tectonic --keep-logs main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.xdv mainNotes.bib
	rm -f figures/*.png

.PHONY: figures pdf clean
