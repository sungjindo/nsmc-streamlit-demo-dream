from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline


def get_nsmc_pipeline(
    model_name_or_path="monologg/koelectra-v3-small-nsmc",
    device= 0,
    
):
    """NSMC Pipeline
    
    Args:
        model_name_or_path (str, optional)
            model name (huggingface.co) or local path,
            Defaults to "monologg/koelectra-v3-small-nsmc",
        device (int, optional):
            -1 for cpu, 0>= for gpu (gpu id),
            Defaults to 0,
            
        Returns:
            TextClassificationPipeline
    """
        

    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=device,
)

    return pipe