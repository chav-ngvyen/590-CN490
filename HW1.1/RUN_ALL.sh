ScriptLoc=${PWD}

cd HomeworkCodes
for i in *.py 
do 
	echo "-----------" $i "-----------" 
	python $i  #run all python scripts
	grep "I HAVE WORKED"
done
exit
