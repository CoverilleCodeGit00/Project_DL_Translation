import numpy as np
import evaluate

# Chargement des metriques
bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")
meteor_metric = evaluate.load("meteor")

def preprocess_function(examples, tokenizer, max_length=128):
    """
    Version moderne de la tokenization pour NLLB/HuggingFace.
    Utilise 'text_target' au lieu du contexte manager as_target_tokenizer.
    """
    inputs = examples["text_fr"]
    targets = examples["text_en"]
    
    # On tokenize tout d'un coup. Le tokenizer sait deja quelle est la 
    # langue source et cible car on l'a defini a l'initialisation.
    model_inputs = tokenizer(
        text=inputs, 
        text_target=targets, 
        max_length=max_length, 
        truncation=True
    )
    
    return model_inputs

def compute_metrics(eval_preds, tokenizer):
    """
    Calcule les metriques de traduction.
    """
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
        
    # Decodage des predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Remplacement des -100 (ignore index) par du padding pour le decodeur
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Nettoyage
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    
    result_bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result_chrf = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result_meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {
        "bleu": result_bleu["score"],
        "chrf": result_chrf["score"],
        "meteor": result_meteor["meteor"] * 100
    }