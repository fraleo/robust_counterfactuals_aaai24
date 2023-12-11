total_cfx=5
nexp=20


for norm in 1 2
do

for beta in  0 0.1 0.2 0.3 0.4 0.5
do

for gamma in 0.1 0.01 0.001
do
python3.7 main.py german ../datasets/german/ ../models/ ../results/ours_german_norm"$norm"_beta"$beta"_gamma"$gamma".txt ours --total_cfx $total_cfx --nexp $nexp --norm $norm --alpha 1000 --beta $beta --gamma $gamma #--opt

python3.7 main.py diabetes ../datasets/diabetes/ ../models/ ../results/ours_diabetes_norm"$norm"_beta"$beta"_gamma"$gamma".txt ours --total_cfx $total_cfx --nexp $nexp --norm $norm --beta $beta --gamma $gamma #--opt

python3.7 main.py no2 ../datasets/no2/ ../models/ ../results/ours_no2_norm"$norm"_beta"$beta"_gamma"$gamma".txt ours --total_cfx $total_cfx --nexp $nexp --norm $norm --beta $beta --gamma $gamma #--opt

python3.7 main.py spam ../datasets/spam/ ../models/ ../results/ours_spam_norm"$norm"_beta"$beta"_gamma"$gamma".txt ours --total_cfx $total_cfx --nexp $nexp --norm $norm --alpha 1000 --beta $beta --gamma $gamma #--opt

python3.7 main.py news ../datasets/news/ ../models/ ../results/ours_news_norm"$norm"_beta"$beta"_gamma"$gamma".txt ours --total_cfx $total_cfx --nexp $nexp --norm $norm --alpha 1000 --beta $beta --gamma $gamma #--opt

done
done
done

for norm in 1 2
do
python3.7 main.py german ../datasets/german/ ../models/ ../results/dice_german_norm$norm.txt dice --total_cfx $total_cfx --nexp $nexp --norm $norm
python3.7 main.py diabetes ../datasets/diabetes/ ../models/ ../results/dice_diabetes_norm$norm.txt dice --total_cfx $total_cfx --nexp $nexp --norm $norm
python3.7 main.py no2 ../datasets/no2/ ../models/ ../results/dice_no2_norm$norm.txt dice --total_cfx $total_cfx --nexp $nexp --norm $norm
python3.7 main.py spam ../datasets/spam/ ../models/ ../results/dice_spam_norm$norm.txt dice --total_cfx $total_cfx --nexp $nexp --norm $norm
python3.7 main.py news ../datasets/news/ ../models/ ../results/dice_news_norm$norm.txt dice --total_cfx $total_cfx --nexp $nexp --norm $norm

done



