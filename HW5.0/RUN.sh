ScriptLoc=${PWD}
# LogLoc=${PWD/Logs}

# cd $ScriptLoc
# # for i in *.py 
# for i in 02-train.py
# do 
# 	echo "-----------" $i "-----------" 
# 	python $i > cd $LogLoc > $i.txt #run all python scripts 
# 	grep "I HAVE WORKED"
# done
# exit
cd $ScriptLoc
for i in 0*.py
do
	filename=${i%%.*}
	# echo $filename$_log.txt
	echo "---Running---" $i "-----------"
	python $i | tee $ScriptLoc/Logs/$filename$_log.txt
	echo "---Done with---" $i 
	echo "---Saved log in---" $ScriptLoc/Logs/$filename$_log.txt
done
exit