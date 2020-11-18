format :
	./format.sh

test : format
	head -n 10 training/formatted_training/training.txt > training/formatted_training/test_training.txt

clean :
	rm -rf training/formatted_training test/formatted_test
