wget -i raw_data_urls.txt -P ../data/ -w 2

mkdir -p ../data/polish && tar -xvf mls_polish_opus.tar.gz ../data/polish/