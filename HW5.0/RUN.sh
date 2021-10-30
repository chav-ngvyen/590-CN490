ScriptLoc=${PWD}

cd $ScriptLoc
for i in 0*.py
do
	filename=${i%%.*}
	echo "---Running---" $i "-----------"
	python $i | tee $ScriptLoc/Logs/$filename$_log.txt
	echo "---Done with---" $i 
	echo "---Saved log in---" $ScriptLoc/Logs/$filename$_log.txt
done
exit