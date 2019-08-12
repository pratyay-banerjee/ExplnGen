
## To train BERT for Explanation Generation:
1. Prepare dataset using the prepare_datasets.py script
2. We modify HuggingFace's pytorch-pretrained-bert's sentence classification scripts for training BERT for this task. The training parameters are in the scripts.

## To run Iterative Re-ranking:
1. You will need Spacy, as we use Spacy word cosine similarity
2. The iterative_reranking_v3.py contains the reranking algorithm. 
```
python iterative_reranking_v3.py --gold ../../worldtree_corpus_textgraphs2019sharedtask_withgraphvis/questions/ARC-Elementary+EXPL-Dev.tsv --eval --expfname ../../data/explanations.csv --args.fname outpreds.txt --scorefname $PATHTOBERTSCORE 
```
3. Download the pretrained model here : DropBoxLink (will update this).