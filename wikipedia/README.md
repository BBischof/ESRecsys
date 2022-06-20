Installation instructions
=========================

Python virtual environment setup

To setup the wikipedia virutal environment, in the parent directory above this one run:

python3 -m venv wikipedia

cd wikipedia

source bin/activate

pip install -r requirements.txt

Additional packages:

java to run spark so make sure that a java runtime is available.
protobuf-compiler to compile the procol buffers if you ever update them like this

protoc  wikipedia.proto --proto_path=../proto  --python_out=.

Getting the Data
================

To get the data that we work with head over to dumps.wikimedia.org/enwiki/202206061/
The data that we used for the book example was from 20220601

https://dumps.wikimedia.org/enwiki/20220601/enwiki-20220601-pages-articles-multistream.xml.bz2

put this file into a download directory, say $DOWNLOAD

Converting the data
===================

The next step is to convert the data into a protocol buffer format, it should take about 10 minutes to process the 19G or so file.

time python3 xml2proto.py --input_file=$DOWNLOAD/enwiki-20220601-pages-articles-multistream.xml.bz2 --output_file=data/enwiki-latest-parsed

After this step, it should all be parallelizable in pyspark. The XML parsing is mostly serial so this step
is the only serial one and the following steps are all parallel in pyspark.

You can view the protocol buffer using the following command:

python3 codex.py --input_file=data/enwiki-latest.pb.bz2 | less

PySpark pipeline
================

Tokenize the documents

bin/spark-submit tokenize_wiki_pyspark.py --input_file=data/enwiki-latest.pb.bz2 --output_file=data/enwiki-latest-tokenized

Make the token and title dictionaries

bin/spark-submit make_dictionary.py \
--input_file="/home/hector/data//wikipedia_tokenized/enwiki-latest-doc/part-?????.bz2" \
--title_output=/home/hector/data/dictionaries/title.tstat.pb.b64.bz2 \
--token_output=/home/hector/data/dictionaries/token.tstat.pb.b64.bz2 \
--min_token_frequency=50 --min_title_frequency=5 \
--max_token_dictionary_size=500000 --max_title_dictionary_size=5000000

Make the co-occurrence for text tokens

bin/spark-submit make_cooccurrence.py \
--input_file="/home/hector/data/wikipedia_tokenized/enwiki-latest-doc/part-?????.bz2" \
--token_dictionary ~/data/dictionaries/token.tstat.pb.b64.bz2 \
--output_file /home/hector/data/wikipedia_cooccur --context_window 10

Train the word embeddings

python train_cooccurence.py \
--train_input_pattern="/home/hector/data/wikipedia_cooccur/part-?????.bz2" \
--validation_input_pattern="/home/hector/data/wikipedia_cooccur/part-????0.bz2" \
--token_dictionary /home/hector/data/dictionaries/token.tstat.pb.b64.bz2 \
--max_terms 20 \
--terms "news,apple,computer,physics,neural,democracy,singapore" \
--tensorboard_dir ~/data/wikipedia_training/logs \
--embedding_dim 64 --batch_size=65536 \
--steps_per_epoch=2000 --validation_steps=100 \
--checkpoint_dir ~/data/wikipedia_training/glove.{epoch:05d}-{val_loss:.5f}.hdf5 --num_epochs=20 \
--learning_rate=0.01

Make the txt2url and url2url training data

bin/spark-submit --master local[4] make_sparse_doc.py \
--input_file="/home/hector/data/wikipedia_tokenized/enwiki-latest-doc/part-?????.bz2" \
--token_dictionary ~/data/dictionaries/token.tstat.pb.b64.bz2 \
--title_dictionary ~/data/dictionaries/title.tstat.pb.b64.bz2 \
--output_txt2url /home/hector/data/wikipedia_txt2url \
--output_url2url /home/hector/data/wikipedia_url2url

Make the co-occurrence for url2url

bin/spark-submit --master local[4] make_dice.py  \
--input_file="${HOME}/data/wikipedia_url2url/part-?????.bz2" \
--output_file="${HOME}/data/wikipedia_url2url_dice"

Dump the url2url dice correlation coefficients

python dump_dice.py --input_file "${HOME}/data/wikipedia_url2url_dice/part-00000.bz2" \
--title_dictionary "${HOME}/data/dictionaries/title.tstat.pb.b64.bz2"


Train the txt2url data

python train_txt2url.py \
--url2url_train_input_pattern "${HOME}/data/wikipedia_url2url_dice/part-????[1-9].bz2" \
--txt2url_train_input_pattern "${HOME}/data/wikipedia_txt2url/part-????[1-9].bz2" \
--url2url_validation_input_pattern "${HOME}/data/wikipedia_url2url_dice/part-????0.bz2" \
--txt2url_validation_input_pattern "${HOME}/data/wikipedia_txt2url/part-????0.bz2" \
--token_dictionary "${HOME}/data/dictionaries/token.tstat.pb.b64.bz2" \
--word_embedding "${HOME}/data/dictionaries/glove.00020-0.00392.hdf5" \
--title_dictionary "${HOME}/data/dictionaries/title.tstat.pb.b64.bz2" \
--tensorboard_dir "${HOME}/data/txt2url_train/logs" \
--checkpoint_dir  "${HOME}/data/txt2url_train/txt2url.hdf5" \
--rnn_size 64 \
--max_sentence_per_example 5 \
--sentence_length 32 \
--url_embedding_dim 64 \
--steps_per_epoch 4000 --validation_steps 100 \
--shuffle_buffer_size 10000 \
--batch_size 1024 \
--learning_rate 0.001 \
--learning_rate_decay 0.8 \
--url_max_norm 5.0 \
--margin 0.1 \
--text_l2 0.01 \
--loss_type "MSE" \
--sentence_csv="deep sea, deep learning, funny operas, edible grasses, turn turn turn,united states constitution,Anarchism is often considered a far-left ideology"



NLP Pipeline
============

Download wikipedia XML dump from https://dumps.wikimedia.org/enwiki/20190301/

Convert XML to Wikipedia proto

python xml2proto.py -i foo.bz2 -o foo.pb.b64.bz2

To dump the wikipedia proto

python codex.py --input foo.pb.b64.bz2

Convert Wikipedia proto to co-occurrence of title and tokenized text

python tokenize_wiki.py -input_file ~/data/wikipedia_proto/enwiki-20190301-pages-articles-multistream1.xml-p10p30302.pb.b64.bz2 --output_file ~/data/wikipedia_proto/enwiki-20190301-pages-articles-multistream1.xml-p10p30302.text_doc.pb.b64.bz2 --stopwords_file=en_wikipedia.stopwords.txt

To dump the tokenized

python codex.py --input_file ~/data/wikipedia_proto/enwiki-20190301-pages-articles-multistream1.xml-p10p30302.text_doc.pb.b64.bz2 --proto doc

Convert tokenized text to title and token dictionary

python make_dictionary.py --input_file ~/data/wikipedia_proto/enwiki-20190301-pages-articles-multistream1.xml-p10p30302.text_doc.pb.b64.bz2 --title_output ~/data/wikipedia_proto/title.dict.pb.b64.bz2 --token_output ~/data/wikipedia_proto/token.dict.pb.b64.bz2 --min_term_frequency 20

To dump the dictionary

python codex.py --input_file ~/data/wikipedia_proto/title.dict.pb.b64.bz2 --proto tstat

To make the English token co-occurrence matrix 

python make_cooccurrence.py --input_file ~/data/wikipedia_proto/enwiki-20190301-pages-articles-multistream1.xml-p10p30302.text_doc.pb.b64.bz2 --token_dictionary ~/data/wikipedia_proto/token.dict.pb.b64.bz2 --output_file ~/data/wikipedia_proto/token.cooccur.pb.b64.bz2

Train the co-occurence matrix embeddings

python train_cooccurence.py --input_file ~/data/wikipedia_proto/token.cooccur.pb.b64.bz2 --token_dictionary ~/data/wikipedia_proto/token.dict.pb.b64.bz2 --max_terms 20 --terms "news,apple,computer,physics,neural,democracy" --tensorboard_dir ~/data/wikipedia_training/logs --embedding_dim 64 --batch_size=65536 --steps_per_epoch=1000 --validation_steps=300 --checkpoint_dir ~/data/wikipedia_training/glove.{epoch:05d}-{val_loss:.5f}.hdf5 --num_epochs=10 
