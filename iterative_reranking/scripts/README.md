
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



## MAP Scores distribution pre and post reranking 
| Param          | PreReranking | ReRanked till 15  | 
|----------------|--------------|-------------------|
| MAP            |0.3890813483823633 |0.42357975830537303| 
| MAP @ 1        |0.6567460317460317 |0.7192460317460317 |                    
| 2              |0.394397201729211  |0.4491287356274116 |                    
| 3              |0.5028164571337207 |0.5407061217199312 |                  
| 4              |0.3733741110589279 |0.38328724780178597|                  
| 5              |0.4005388750681937 |0.4412143551037883 |                  
| 6              |0.3474970488931572 |0.3634400308835687 |                  
| 7              |0.38724720718255295|0.4164468343195992 |                  
| 8              |0.38742025592657   |0.44355025820864924|                  
| 9              |0.24177613632910175|0.26488851906357075|                  
| 10             |0.28589244236553696|0.32328098342415956|                  
| 11             |0.27569807308556743|0.30105743928654594|                  
| 12             |0.2402659407419108 |0.3216624785774602 |                  
| 13             |0.1726643148547781 |0.16416995658135752|                 
| 14             |0.34512989459281845|0.3565808552604439 |                  
| 15             |0.19346145550717087|0.2353926724383878 |                  
| 16             |0.21321315333294488|0.20987322448169635|                  
| Roles           |
| CENTRAL        |0.35892151785888   |0.39116299016334183|                  
| GROUNDING      |0.06305258286821085|0.09650284304004954|                  
| LEXGLUE        |0.17208501107078403|0.15372167362560069|                  
| NE             |7.410692609871888e-06|7.410692609871888e-06|                  
| NEG            |0.0003050999241296848|0.000586038258165364|                  
| BACKGROUND     |0.025301026905644373|0.022599061163595342|                  
