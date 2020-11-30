format :
	./format.sh

test : format
	head -n 10 training/formatted_train/train.txt > train/formatted_train/test_train.txt

clean :
	rm -rf test.txt train.txt tune.txt
