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
for ln in merged_snd-cert_test.txt
do
  while read ln; do

    count=$((count+1))
    echo $count
    # echo $ln | java -jar negsel2.jar -alphabet file://english.train -self english.train -n 10 -r 4 -c -l >> test_results_4.txt
    echo $ln | java -jar negsel2.jar -alphabet file://snd-cert.alpha -self snd-cert.train -n 7 -r 4 -c -l >> test_results_cert_7+4.txt
  done <merged_snd-cert_test.txt
done
echo $count

# varname="fall_me_is"
# echo $varname | java -jar negsel2.jar -alphabet file://english.train -self english.train -n 10 -r 4 -c -l
