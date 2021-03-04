# ! /bin/bash
# for ((a=022; a <= 658 ; a++))
# do
#    printf "./filter filename%04d.bmp outputfile lower upper" $a | "sh"
# done
# echo "Reached"
# printf "java -jar negsel2.jar -alphabet file://english.train -self english.train -n 10 -r 4 -c -l"
echo "Start"
count=0


# for ln in merged.test
for ln in merged.test
do
  while read ln; do
    # echo "${ln}"
    count=$((count+1))
    echo $count
    echo $ln | java -jar negsel2.jar -alphabet file://english.train -self english.train -n 10 -r 4 -c -l >> test_results_4.txt
  done <merged.test
done
echo $count

# varname="fall_me_is"
# echo $varname | java -jar negsel2.jar -alphabet file://english.train -self english.train -n 10 -r 4 -c -l
