import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from transformers import BartTokenizer

from modeling_bart import BartForConditionalGeneration

from longbart import LongformerSelfAttentionForBart

def create_long_model(save_model_to, base_model='bart-large', attention_window=512, max_pos=4096):
    model = BartForConditionalGeneration.from_pretrained(base_model)
    tokenizer = BartTokenizer.from_pretrained('bart-large', model_max_length=max_pos)
    config = model.config

    # in BART attention_probs_dropout_prob is attention_dropout, but LongformerSelfAttention
    # expects attention_probs_dropout_prob, so set it here  
    config.attention_probs_dropout_prob = config.attention_dropout

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.model.encoder.embed_positions.weight.shape
    # config.max_position_embeddings = max_pos
    config.encoder_max_position_embeddings = max_pos
    max_pos += 2  # NOTE: BART has positions 0,1 reserved, so embedding size is max position + 2
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.model.encoder.embed_positions.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.model.encoder.embed_positions.weight[2:]
        k += step
    model.model.encoder.embed_positions.weight.data = new_pos_embed

    # replace the `modeling_bart.SelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.model.encoder.layers):
        longformer_self_attn_for_bart = LongformerSelfAttentionForBart(config, layer_id=i)
        
        longformer_self_attn_for_bart.longformer_self_attn.query = layer.self_attn.q_proj
        longformer_self_attn_for_bart.longformer_self_attn.key = layer.self_attn.k_proj
        longformer_self_attn_for_bart.longformer_self_attn.value = layer.self_attn.v_proj

        longformer_self_attn_for_bart.longformer_self_attn.query_global = layer.self_attn.q_proj
        longformer_self_attn_for_bart.longformer_self_attn.key_global = layer.self_attn.k_proj
        longformer_self_attn_for_bart.longformer_self_attn.value_global = layer.self_attn.v_proj

        longformer_self_attn_for_bart.output = layer.self_attn.out_proj

        layer.self_attn = longformer_self_attn_for_bart

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer