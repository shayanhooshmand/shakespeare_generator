#!/bin/sh
cd training
ls *.xml | while read FILE
do
	cat $FILE | grep "<LINE>" | sed 's/<LINE>//g' | sed 's/<\/LINE>//g' | sed 's/<STAGEDIR>//g' | sed 's/<\/STAGEDIR>//g' >> ../train.txt
done

cd ../tune
ls *.xml | while read FILE
do
        cat $FILE | grep "<LINE>" | sed 's/<LINE>//g' | sed 's/<\/LINE>//g' | sed 's/<STAGEDIR>//g' | sed 's/<\/STAGEDIR>//g' >> ../tune.txt
done

cd ../test
ls *.xml | while read FILE
do
	cat $FILE | grep "<LINE>" | sed 's/<LINE>//g' | sed 's/<\/LINE>//g' | sed 's/<STAGEDIR>//g' | sed 's/<\/STAGEDIR>//g' >> ../test.txt

done
