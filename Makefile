report.pdf: report.tex table.tex K-MeansPlusPlus.png K-meansParallel.png K-meansParallel_MC.png
	pdflatex report
	pdflatex report
	pdflatex report
	pdflatex report
	pdflatex report

table.tex: SimData.csv
	python prepare_result.py

K-MeansPlusPlus.png: SimData.csv
	python KMeansPlusPlus.py
	python prepare_result.py

K-meansParallel.png: SimData.csv
	python KMeansParallel.py
	python prepare_result.py

K-meansParallel_C.png: SimData.csv
	python Main.ipynb

K-meansParallel_MC.png: SimData.csv
	python KMeansParallel_MC.py
	python prepare_result.py

SimData.csv: 
	python SimData.py

.PHONY: all clean

all: report.pdf

clean:
	rm -f *csv *png *aux *log *png table.tex *pytxcode *pdf
