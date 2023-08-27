import ivy
from ivy_models.base import BaseModel
from ivy_models.helpers import load_transformers_weights
from .layers import AlbertAttention, AlbertFeedForward, AlbertEmbedding, AlbertConfig



# Albert Model Construction
def apply_chunking_to_forward(
    feed_forward_module, chunk_size: int, chunk_dim: int, *input_tensors
):
    if chunk_size is not None and chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )
        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(
            ivy.split(input_tensor, num_or_size_splits=num_chunks, axis=chunk_dim)
            for input_tensor in input_tensors
        )
        # apply forward fn to every tuple
        output_chunks = tuple(
            feed_forward_module(*input_tensors_chunk)
            for input_tensors_chunk in zip(*input_tensors_chunks)
        )
        # concatenate output at same dimension
        return ivy.concat(output_chunks, axis=chunk_dim)

    return feed_forward_module(*input_tensors)


class AlbertLayer(ivy.Module):
    def __init__(self, config: AlbertConfig, v=None):
        self.config = config
        self.chunk_size = config.chunk_size_feed_forward
        self.is_deocder = config.is_decoder
        super(AlbertLayer, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.attention = AlbertAttention(**self.config.get_attn_attrs())
        self.ffd = AlbertFeedForward(**self.config.get_ffd_attrs())

    def _forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        outputs = self.attention(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

        ffd_out = apply_chunking_to_forward(self.ffd, self.chunk_size, 1, outputs[0])
        return (ffd_out,) + outputs[1:]


class AlbertEncoder(ivy.Module):
    def __init__(self, config: AlbertConfig, v=None):
        self.config = config
        super(AlbertEncoder, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.layer = [
            AlbertLayer(self.config) for _ in range(self.config.num_hidden_layers)
        ]

    def _forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
    ):
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        return hidden_states, all_self_attentions, next_decoder_cache


class AlbertPooler(ivy.Module):
    def __init__(self, config: AlbertConfig, v=None):
        self.config = config
        super(AlbertPooler, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.dense = ivy.Linear(self.config.hidden_size, self.config.hidden_size)
        self.activation = ivy.tanh

    def _forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token [CLS].
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class AlbertModel(BaseModel):
    def __init__(self, config: AlbertConfig, pooler_out=False, v=None):
        self.config = config
        self.pooler_out = pooler_out
        super(AlbertModel, self).__init__(v=v)

    @classmethod
    def get_spec_class(self):
        return AlbertConfig

    def _build(self, *args, **kwargs):
        self.embeddings = AlbertEmbedding(**self.config.get_embd_attrs())
        self.encoder = AlbertEncoder(self.config)
        self.pooler = AlbertPooler(self.config) if self.pooler_out else None

    def _forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
    ):
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )
        embeddings = self.embeddings(
            input_ids,
            token_type_ids,
            position_ids,
            past_key_values_length,
        )

        encoder_outs = self.encoder(
            embeddings,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
        )
        if self.pooler_out:
            pooler_out = self.pooler(encoder_outs[0])
        else:
            pooler_out = None
        return {
            "pooler_output": pooler_out,
            "last_hidden_state": encoder_outs[0],
            "attention_probs": encoder_outs[1],
            "next_decoder_cache": encoder_outs[2],
        }


# Mapping and loading section
def _albert_weights_mapping(name):
    key_map = [(f"__v{i}__", f"__{j}__") for i, j in zip(range(12), range(12))]
    key_map = key_map + [
        ("attention__dense", "attention.output.dense"),
        ("attention__LayerNorm", "attention.output.LayerNorm"),
    ]
    key_map = key_map + [
        ("ffd__dense1", "intermediate.dense"),
        ("ffd__dense2", "output.dense"),
        ("ffd__LayerNorm", "output.LayerNorm"),
    ]
    name = name.replace("__w", ".weight").replace("__b", ".bias")

    for ref, new in key_map:
        name = name.replace(ref, new)
    name = name.replace("__", ".")
    return name


def albert_base_v1(pretrained=True):
    # instantiate the hyperparameters same as albert
    # set the dropout rate to 0.0 to avoid stochasticity in the output
    config = AlbertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout=0.0,
        attn_drop_rate=0.0,
        max_position_embeddings=512,
    )
    model = AlbertModel(config, pooler_out=True)
    if pretrained:
        w_clean = load_transformers_weights(
            "albert-base-v1", model, _albert_weights_mapping
        )
        model.v = w_clean
    return model
