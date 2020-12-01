evaluate : format
	python3 evaluate.py

interactive : format
	python3 -i ngram_model.py

format :
	./format.sh

clean :
	rm -rf test.txt train.txt tune.txt
