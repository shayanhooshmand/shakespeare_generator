#!/bin/sh
cd training
mkdir formatted_training
ls *.xml | while read FILE
do
	cat $FILE | grep "<LINE>" | sed 's/<LINE>//g' | sed 's/<\/LINE>//g' | sed 's/<STAGEDIR>//g' | sed 's/<\/STAGEDIR>//g' > formatted_training/$FILE
done

cd ../test
mkdir formatted_test
ls *.xml | while read FILE
do
	cat $FILE | grep "<LINE>" | sed 's/<LINE>//g' | sed 's/<\/LINE>//g' | sed 's/<STAGEDIR>//g' | sed 's/<\/STAGEDIR>//g' > formatted_test/$FILE

done
