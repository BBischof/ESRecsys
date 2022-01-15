Installation instructions
=========================

Python virtual environment setup

To setup the wikipedia virutal environment, in the toy directory run:

python3 -m venv wikipedia

cd wikipedia

source bin/activate

pip install -r requirements.txt

If Google drive filestream is used, the data is also in

/Volumes/GoogleDrive/Team\ Drives/Engineering/corpora/wikipedia//wikipedia_proto

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