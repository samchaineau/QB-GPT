import tensorflow as tf
from typing import List, Optional, Union
import numpy as np

def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.

    Returns:
        `List[int]`: The shape of the tensor as a list.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

class PlayTypeEncoder(tf.keras.Model):
  def __init__(self, vocab_size : int, embedding_dim : int):
        super(PlayTypeEncoder, self).__init__()

        self.Embedding = tf.keras.layers.Embedding(input_dim = vocab_size,
                                                   output_dim = embedding_dim)

  def call(self, x):
    embed = self.Embedding(x["PlayType"])
    return embed

class PositionEncoder(tf.keras.Model):
  def __init__(self, vocab_size : int, embedding_dim : int):
        super(PositionEncoder, self).__init__()

        self.Embedding = tf.keras.layers.Embedding(input_dim = vocab_size,
                                                   output_dim = embedding_dim)

  def call(self, x):
    embed = self.Embedding(x["position_ids"])
    return embed

class ScrimmageEncoder(tf.keras.Model):
  def __init__(self, vocab_size : int, embedding_dim : int):
        super(ScrimmageEncoder, self).__init__()

        self.Embedding = tf.keras.layers.Embedding(input_dim = vocab_size,
                                                   output_dim = embedding_dim)

  def call(self, x):
    embed = self.Embedding(x["scrim_ids"])
    return embed

class StartEncoder(tf.keras.Model):
  def __init__(self, vocab_size : int, embedding_dim : int):
        super(StartEncoder, self).__init__()

        self.Embedding = tf.keras.layers.Embedding(input_dim = vocab_size,
                                                   output_dim = embedding_dim)

  def call(self, x):
    embed = self.Embedding(x["start_ids"])
    return embed

class OffDefEncoder(tf.keras.Model):
  def __init__(self, vocab_size : int, embedding_dim : int):
        super(OffDefEncoder, self).__init__()

        self.Embedding = tf.keras.layers.Embedding(input_dim = vocab_size,
                                                   output_dim = embedding_dim)

  def call(self, x):
    embed = self.Embedding(x["OffDef"])
    return embed

class TypeEncoder(tf.keras.Model):
  def __init__(self, vocab_size : int, embedding_dim : int):
        super(TypeEncoder, self).__init__()

        self.Embedding = tf.keras.layers.Embedding(input_dim = vocab_size,
                                                   output_dim = embedding_dim)

  def call(self, x):
    embed = self.Embedding(x["token_type_ids"])
    return embed

class PositionalEncoder(tf.keras.Model):
  def __init__(self, vocab_size : int, embedding_dim : int):
        super(PositionalEncoder, self).__init__()

        self.Embedding = tf.keras.layers.Embedding(input_dim = vocab_size,
                                                   output_dim = embedding_dim)

  def call(self, x):
    embed = self.Embedding(x["pos_ids"])
    return embed

class InputEncoder(tf.keras.Model):
  def __init__(self, vocab_size : int, embedding_dim : int):
        super(InputEncoder, self).__init__()

        self.Embedding = tf.keras.layers.Embedding(input_dim = vocab_size,
                                                   output_dim = embedding_dim)

  def call(self, x):
    embed = self.Embedding(x["input_ids"])
    return embed

class Embedding(tf.keras.Model):
  def __init__(self,
               input_vocab_size : int,
               positional_vocab_size : int,
               position_vocab_size : int,
               scrimmage_vocab_size : int,
               start_vocab_size: int,
               offdef_vocab_size : int,
               type_vocab_size : int,
               playtype_vocab_size : int,
               embedding_dim : int):
        super(Embedding, self).__init__()

        self.InputEmbedding = InputEncoder(vocab_size=input_vocab_size,
                                           embedding_dim=embedding_dim)
        self.PositionalEmbedding = PositionalEncoder(vocab_size=positional_vocab_size,
                                                     embedding_dim=embedding_dim)
        self.PositionEmbedding = PositionEncoder(vocab_size=position_vocab_size,
                                                     embedding_dim=embedding_dim)
        self.ScrimEmbedding = ScrimmageEncoder(vocab_size=scrimmage_vocab_size,
                                                     embedding_dim=embedding_dim)
        self.StartEmbedding = StartEncoder(vocab_size=start_vocab_size,
                                                     embedding_dim=embedding_dim)
        self.OffDefEmbedding = OffDefEncoder(vocab_size=offdef_vocab_size,
                                             embedding_dim=embedding_dim)
        self.TypeEmbedding = TypeEncoder(vocab_size=type_vocab_size,
                                             embedding_dim=embedding_dim)
        self.PlayTypeEmbedding = PlayTypeEncoder(vocab_size=playtype_vocab_size,
                                                 embedding_dim=embedding_dim)
        self.Add = tf.keras.layers.Add()

  def call(self, x):
    input_embed = self.InputEmbedding(x)
    positional_embed = self.PositionalEmbedding(x)
    position_embed = self.PositionEmbedding(x)
    scrim_embed = self.ScrimEmbedding(x)
    start_embed = self.StartEmbedding(x)
    type_embed = self.TypeEmbedding(x)
    offdef_embed = self.OffDefEmbedding(x)
    playtype_embed = self.PlayTypeEmbedding(x)

    embed = self.Add([input_embed,
                      positional_embed,
                      position_embed,
                      scrim_embed,
                      start_embed,
                      type_embed,
                      offdef_embed,
                      playtype_embed])

    return embed

class Transformers(tf.keras.Model):
  def __init__(self,
               num_heads : int,
               hidden_dim : int,
               output_dim : int,
               diag_masks : bool):
        super(Transformers, self).__init__()
        
        self.diag_masks = diag_masks
        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_dim
        self.total_dim = num_heads * hidden_dim
        self.output_dim = output_dim
        
        self.NormIn = tf.keras.layers.LayerNormalization(name = "Norm_in")
        self.Query = tf.keras.layers.Dense(self.total_dim, name = "Query", use_bias = False)
        self.Key = tf.keras.layers.Dense(self.total_dim, name = "Key", use_bias = False)
        self.Value = tf.keras.layers.Dense(self.total_dim, name = "Value", use_bias = False)
        
        self.DenseAtt = tf.keras.layers.Dense(hidden_dim, activation = "relu", use_bias = False)
        
        self.Add = tf.keras.layers.Add(name = "Add")
        self.Drop = tf.keras.layers.Dropout(rate = 0.1)
        
        self.DenseOut = tf.keras.layers.Dense(output_dim, name = "Dense", activation = "relu")
        self.NormOut = tf.keras.layers.LayerNormalization(name = "Norm_out")

  def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

  def create_causal_masks(self, temp_ids):
      # Use broadcasting to create the 2D comparison tensor
      causal_mask = temp_ids[:, :, tf.newaxis] >= temp_ids[:, tf.newaxis, :]
      causal_mask = (tf.cast(causal_mask, dtype=tf.float32) - 1) * 1000000
      reshaped_tensor = tf.expand_dims(causal_mask, axis=1)
      duplicated_tensor = tf.tile(reshaped_tensor, multiples=[1, self.num_attention_heads, 1, 1])
      return duplicated_tensor
    
  def create_diag_masks(self, hidden_state):
    dims = shape_list(hidden_state)
    matrix = tf.linalg.diag(tf.ones((dims[0], dims[1], dims[2]), dtype=tf.float32))
    return matrix*-1000000

  def create_attention_mask(self, attn_mask):
    attn_mask = (tf.cast(attn_mask, dtype=tf.float32) -1) * 1000000
    reshaped_tensor = tf.expand_dims(attn_mask, axis=1)
    reshaped_tensor = tf.expand_dims(reshaped_tensor, axis=1)
    duplicated_tensor = tf.tile(reshaped_tensor, multiples=[1, self.num_attention_heads, 1, 1])
    return duplicated_tensor

  def compute_scaled_attn_scores(self, query, key):
    attention_scores = tf.matmul(query, key, transpose_b=True)  # Transpose the second sequence

    # If you want scaled dot-product attention, divide by the square root of the embedding dimension
    embedding_dim = query.shape[-1]
    scaled_attention_scores = attention_scores / tf.math.sqrt(tf.cast(embedding_dim, dtype=tf.float32))

    return scaled_attention_scores

  def compute_attention_weigths(self, query, key, temp_ids, masks):

    attn_masks = self.create_attention_mask(masks)
    causal_masks = self.create_causal_masks(temp_ids)
    scaled_attn_scores = self.compute_scaled_attn_scores(query, key)
    if self.diag_masks == True:
      diag_masks = self.create_diag_masks(query)
      attn_scores = scaled_attn_scores + attn_masks + causal_masks + diag_masks
    else:
      attn_scores = scaled_attn_scores + attn_masks + causal_masks + diag_masks
      
    return tf.nn.softmax(attn_scores, axis = -1)

  def get_preds_and_attention(self,
           embeddings,
           temporal_ids,
           attention_masks):

    query = self.Query(embeddings)
    key = self.Key(embeddings)
    value = self.Value(embeddings)

    attention_weights = self.compute_attention_weigths(query, key, temporal_ids, attention_masks)

    attention_scores = tf.matmul(attention_weights, value)
    attention_scores = self.Dense(attention_scores)

    output = self.Add([attention_scores, embeddings])
    output = self.Drop(output)
    output = self.Norm(output)
    return output, attention_weights

  def call(self,
           hidden_states : tf.Tensor,
           temporal_ids,
           attention_masks):

    batch_size = shape_list(hidden_states)[0]
    
    norm_hidden_states = self.NormIn(hidden_states)
    
    query = self.Query(norm_hidden_states)
    queries = self.transpose_for_scores(query, batch_size)

    key = self.Key(norm_hidden_states)
    keys = self.transpose_for_scores(key, batch_size)

    value = self.Value(norm_hidden_states)
    values = self.transpose_for_scores(value, batch_size)

    attention_weights = self.compute_attention_weigths(queries, keys, temporal_ids, attention_masks)
    attention_scores = tf.matmul(attention_weights, values)
    attention_scores = tf.transpose(attention_scores, perm=[0, 2, 1, 3])
    attention_scores = tf.reshape(tensor=attention_scores, shape=(batch_size, -1, self.total_dim))
    attention_scores = self.DenseAtt(attention_scores)
    
    output = self.Add([attention_scores, hidden_states])
    norm_output = self.NormOut(output)
    
    densed_output = self.DenseOut(norm_output)
    output = self.Add([densed_output, output])
    output = self.Drop(output)
    return output

class Encoder(tf.keras.Model):
  def __init__(self,
               input_vocab_size : int,
               positional_vocab_size : int,
               position_vocab_size : int,
               scrimmage_vocab_size : int,
               start_vocab_size: int,
               offdef_vocab_size : int,
               type_vocab_size : int,
               playtype_vocab_size : int,
               embedding_dim : int,
               hidden_dim : int,
               num_heads : int,
               diag_masks : bool):
        super(Encoder, self).__init__()
        
        self.num_heads = num_heads
        self.diag_masks = diag_masks
        self.Embedding = Embedding(input_vocab_size = input_vocab_size,
                                   positional_vocab_size = positional_vocab_size,
                                   position_vocab_size = position_vocab_size,
                                   scrimmage_vocab_size = scrimmage_vocab_size,
                                   start_vocab_size = start_vocab_size,
                                   type_vocab_size = type_vocab_size,
                                   offdef_vocab_size = offdef_vocab_size,
                                   playtype_vocab_size = playtype_vocab_size,
                                   embedding_dim = embedding_dim)

        self.Attention1 = Transformers(num_heads = self.num_heads,
                                       hidden_dim = hidden_dim,
                                       output_dim = embedding_dim, 
                                       diag_masks = self.diag_masks)

  def call(self,
           x):

    embed = self.Embedding(x)
    h1 = self.Attention1(embed, x["pos_ids"], x["attention_mask"])

    return h1

class EncoderL(tf.keras.Model):
  def __init__(self,
               input_vocab_size : int,
               positional_vocab_size : int,
               position_vocab_size : int,
               scrimmage_vocab_size : int,
               start_vocab_size: int,
               offdef_vocab_size : int,
               type_vocab_size : int,
               playtype_vocab_size : int,
               embedding_dim : int,
               hidden_dim : int,
               num_heads : int,
               diag_masks : bool):
        super(EncoderL, self).__init__()
        
        self.num_heads = num_heads
        self.diag_masks = diag_masks
        self.Embedding = Embedding(input_vocab_size = input_vocab_size,
                                   positional_vocab_size = positional_vocab_size,
                                   position_vocab_size = position_vocab_size,
                                   scrimmage_vocab_size = scrimmage_vocab_size,
                                   start_vocab_size = start_vocab_size,
                                   type_vocab_size = type_vocab_size,
                                   offdef_vocab_size = offdef_vocab_size,
                                   playtype_vocab_size = playtype_vocab_size,
                                   embedding_dim = embedding_dim)

        self.Attention1 = Transformers(num_heads = self.num_heads,
                                       hidden_dim = hidden_dim,
                                       output_dim = embedding_dim, 
                                       diag_masks = self.diag_masks)
        self.Attention2 = Transformers(num_heads = self.num_heads,
                                       hidden_dim = hidden_dim,
                                       output_dim = embedding_dim, 
                                       diag_masks = self.diag_masks)

  def call(self,
           x):

    embed = self.Embedding(x)
    h1 = self.Attention1(embed, x["pos_ids"], x["attention_mask"])
    h2 = self.Attention2(h1, x["pos_ids"], x["attention_mask"])

    return h2

class EncoderXL(tf.keras.Model):
  def __init__(self,
               input_vocab_size : int,
               positional_vocab_size : int,
               position_vocab_size : int,
               scrimmage_vocab_size : int,
               start_vocab_size: int,
               offdef_vocab_size : int,
               type_vocab_size : int,
               playtype_vocab_size : int,
               embedding_dim : int,
               hidden_dim : int,
               num_heads : int,
               diag_masks : bool):
        super(EncoderXL, self).__init__()
        
        self.num_heads = num_heads
        self.diag_masks = diag_masks
        self.Embedding = Embedding(input_vocab_size = input_vocab_size,
                                   positional_vocab_size = positional_vocab_size,
                                   position_vocab_size = position_vocab_size,
                                   scrimmage_vocab_size = scrimmage_vocab_size,
                                   start_vocab_size = start_vocab_size,
                                   type_vocab_size = type_vocab_size,
                                   offdef_vocab_size = offdef_vocab_size,
                                   playtype_vocab_size = playtype_vocab_size,
                                   embedding_dim = embedding_dim)

        self.Attention1 = Transformers(num_heads = self.num_heads,
                                       hidden_dim = hidden_dim,
                                       output_dim = embedding_dim, 
                                       diag_masks = self.diag_masks)
        self.Attention2 = Transformers(num_heads = self.num_heads,
                                       hidden_dim = hidden_dim,
                                       output_dim = embedding_dim, 
                                       diag_masks = self.diag_masks)
        self.Attention3 = Transformers(num_heads = self.num_heads,
                                       hidden_dim = hidden_dim,
                                       output_dim = embedding_dim, 
                                       diag_masks = self.diag_masks)

  def call(self,
           x):

    embed = self.Embedding(x)
    h1 = self.Attention1(embed, x["pos_ids"], x["attention_mask"])
    h2 = self.Attention2(h1, x["pos_ids"], x["attention_mask"])
    h3 = self.Attention3(h2, x["pos_ids"], x["attention_mask"])

    return h3

class QBGPT(tf.keras.Model):
  def __init__(self,
               input_vocab_size : int,
               positional_vocab_size : int,
               position_vocab_size : int,
               scrimmage_vocab_size : int,
               start_vocab_size: int,
               offdef_vocab_size : int,
               type_vocab_size : int,
               playtype_vocab_size : int,
               embedding_dim : int,
               hidden_dim : int,
               num_heads : int,
               diag_masks : bool,
               to_pred_size : int):
        super(QBGPT, self).__init__()

        self.Encoder = Encoder(input_vocab_size = input_vocab_size,
                               positional_vocab_size = positional_vocab_size,
                               position_vocab_size = position_vocab_size,
                               scrimmage_vocab_size = scrimmage_vocab_size,
                               start_vocab_size = start_vocab_size,
                               type_vocab_size = type_vocab_size,
                               offdef_vocab_size = offdef_vocab_size,
                               playtype_vocab_size = playtype_vocab_size,
                               embedding_dim = embedding_dim,
                               hidden_dim = hidden_dim,
                               num_heads = num_heads,
                               diag_masks = diag_masks)

        self.Logits = tf.keras.layers.Dense(to_pred_size, activation = "relu")

  def call(self, x):

    encoded = self.Encoder(x)
    logits = self.Logits(encoded)

    return logits

class LargeQBGPT(tf.keras.Model):
  def __init__(self,
               input_vocab_size : int,
               positional_vocab_size : int,
               position_vocab_size : int,
               scrimmage_vocab_size : int,
               start_vocab_size: int,
               offdef_vocab_size : int,
               type_vocab_size : int,
               playtype_vocab_size : int,
               embedding_dim : int,
               hidden_dim : int,
               num_heads : int,
               diag_masks : bool,
               to_pred_size : int):
        super(LargeQBGPT, self).__init__()

        self.Encoder = EncoderL(input_vocab_size = input_vocab_size,
                               positional_vocab_size = positional_vocab_size,
                               position_vocab_size = position_vocab_size,
                               scrimmage_vocab_size = scrimmage_vocab_size,
                               start_vocab_size = start_vocab_size,
                               type_vocab_size = type_vocab_size,
                               offdef_vocab_size = offdef_vocab_size,
                               playtype_vocab_size = playtype_vocab_size,
                               embedding_dim = embedding_dim,
                               hidden_dim = hidden_dim,
                               num_heads = num_heads,
                               diag_masks = diag_masks)

        self.Logits = tf.keras.layers.Dense(to_pred_size, activation = "relu")

  def call(self, x):

    encoded = self.Encoder(x)
    logits = self.Logits(encoded)

    return logits

class XLargeQBGPT(tf.keras.Model):
  def __init__(self,
               input_vocab_size : int,
               positional_vocab_size : int,
               position_vocab_size : int,
               scrimmage_vocab_size : int,
               start_vocab_size: int,
               offdef_vocab_size : int,
               type_vocab_size : int,
               playtype_vocab_size : int,
               embedding_dim : int,
               hidden_dim : int,
               num_heads : int,
               diag_masks : bool,
               to_pred_size : int):
        super(XLargeQBGPT, self).__init__()

        self.Encoder = EncoderXL(input_vocab_size = input_vocab_size,
                               positional_vocab_size = positional_vocab_size,
                               position_vocab_size = position_vocab_size,
                               scrimmage_vocab_size = scrimmage_vocab_size,
                               start_vocab_size = start_vocab_size,
                               type_vocab_size = type_vocab_size,
                               offdef_vocab_size = offdef_vocab_size,
                               playtype_vocab_size = playtype_vocab_size,
                               embedding_dim = embedding_dim,
                               hidden_dim = hidden_dim,
                               num_heads = num_heads,
                               diag_masks = diag_masks)

        self.Logits = tf.keras.layers.Dense(to_pred_size, activation = "relu")

  def call(self, x):

    encoded = self.Encoder(x)
    logits = self.Logits(encoded)

    return logits