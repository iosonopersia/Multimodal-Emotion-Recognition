mkdir data
cd data
wget https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
tar -xvf MELD.Raw.tar.gz
rm MELD.Raw.tar.gz

cd MELD.Raw
tar -xvf dev.tar.gz
tar -xvf test.tar.gz
tar -xvf train.tar.gz

rm dev.tar.gz
rm test.tar.gz
rm train.tar.gz

cd..
