mkdir data
pushd data

cURL https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz --output MELD.Raw.tar.gz
tar -xvzf MELD.Raw.tar.gz
del MELD.Raw.tar.gz

cd MELD.Raw
tar -xvzf dev.tar.gz
tar -xvzf test.tar.gz
tar -xvzf train.tar.gz

del dev.tar.gz
del test.tar.gz
del train.tar.gz

popd
