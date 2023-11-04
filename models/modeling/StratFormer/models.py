import tensorflow as tf 
import tensorflow_probability as tfp
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

class DownEncoder(tf.keras.Model):
  def __init__(self, vocab_size : int, embedding_dim : int):
        super(DownEncoder, self).__init__()

        self.Embedding = tf.keras.layers.Embedding(input_dim = vocab_size,
                                                   output_dim = embedding_dim)

  def call(self, x):
    embed = self.Embedding(x["down_ID"])
    return embed

class SeasonEncoder(tf.keras.Model):
  def __init__(self, vocab_size : int, embedding_dim : int):
        super(SeasonEncoder, self).__init__()

        self.Embedding = tf.keras.layers.Embedding(input_dim = vocab_size,
                                                   output_dim = embedding_dim)

  def call(self, x):
    embed = self.Embedding(x["season_ID"])
    return embed

class TeamEncoder(tf.keras.Model):
  def __init__(self, vocab_size : int, embedding_dim : int):
        super(TeamEncoder, self).__init__()

        self.Embedding = tf.keras.layers.Embedding(input_dim = vocab_size,
                                                   output_dim = embedding_dim)

  def call(self, x):
    embed = self.Embedding(x["team_ID"])
    return embed

class PlayerEncoder(tf.keras.Model):
  def __init__(self, vocab_size : int, embedding_dim : int):
        super(PlayerEncoder, self).__init__()

        self.Embedding = tf.keras.layers.Embedding(input_dim = vocab_size,
                                                   output_dim = embedding_dim)

  def call(self, x):
    embed = self.Embedding(x["player_ids"])
    return embed


class MetaEmbedding(tf.keras.Model):
  def __init__(self, 
               team_vocab_size : int, 
               player_vocab_size : int, 
               season_vocab_size : int, 
               down_vocab_size : int, 
               embedding_dim : int):
        super(MetaEmbedding, self).__init__()
        
        self.embedding_dim = embedding_dim

        self.TeamEmbedding = TeamEncoder(vocab_size= team_vocab_size,
                                         embedding_dim=embedding_dim)
        self.PlayerEmbedding = PlayerEncoder(vocab_size= player_vocab_size,
                                             embedding_dim=embedding_dim)
        self.SeasonEmbedding = SeasonEncoder(vocab_size= season_vocab_size,
                                             embedding_dim=embedding_dim)
        self.DownEmbedding = DownEncoder(vocab_size= down_vocab_size,
                                         embedding_dim=embedding_dim)
        
        self.Add = tf.keras.layers.Add()

  def call(self, x):
    team_embed = self.TeamEmbedding(x)
    player_embed = self.PlayerEmbedding(x)
    season_embed = self.SeasonEmbedding(x)
    down_embed = self.DownEmbedding(x)
    
    added = self.Add([team_embed, player_embed, season_embed, down_embed])
    
    return added


class SimpleTransformers(tf.keras.Model):
  def __init__(self,
               hidden_dim : int,
               output_dim : int):
        super(SimpleTransformers, self).__init__()

        self.num_attention_heads = 1
        self.attention_head_size = hidden_dim
        self.total_dim = 1 * hidden_dim
        self.output_dim = output_dim
        
        self.NormIn = tf.keras.layers.LayerNormalization(name = "Norm_in")
        self.Query = tf.keras.layers.Dense(self.total_dim, name = "Query", use_bias = False)
        self.Key = tf.keras.layers.Dense(self.total_dim, name = "Key", use_bias = False)
        self.Value = tf.keras.layers.Dense(self.total_dim, name = "Value", use_bias = False)
        
        self.Add = tf.keras.layers.Add(name = "Add")
        self.Drop = tf.keras.layers.Dropout(rate = 0.1)
        
        self.DenseOut = tf.keras.layers.Dense(output_dim, name = "Dense", activation = "relu")
        self.NormOut = tf.keras.layers.LayerNormalization(name = "Norm_out")

  def compute_scaled_attn_scores(self, query, key):
    attention_scores = tf.matmul(query, key, transpose_b=True)  # Transpose the second sequence

    # If you want scaled dot-product attention, divide by the square root of the embedding dimension
    embedding_dim = query.shape[-1]
    scaled_attention_scores = attention_scores / tf.math.sqrt(tf.cast(embedding_dim, dtype=tf.float32))

    return scaled_attention_scores

  def call(self,
           q : tf.Tensor,
           k : tf.Tensor,
           v : tf.Tensor):
    
    norm_hidden_states_q = self.NormIn(q)
    norm_hidden_states_k = self.NormIn(k)
    norm_hidden_states_v = self.NormIn(v)
    
    queries = self.Query(norm_hidden_states_q)
    keys = self.Key(norm_hidden_states_k)
    values = self.Key(norm_hidden_states_v)
    
    attention_weights = self.compute_scaled_attn_scores(queries, keys)
    attention_scores = tf.matmul(attention_weights, values)
    
    output = self.Add([attention_scores, v])
    norm_output = self.NormOut(output)
    
    densed_output = self.DenseOut(norm_output)
    output = self.Add([densed_output, output])
    output = self.Drop(output)
    return output

class StratEncoder(tf.keras.Model):
    def __init__(self,
                 num_spec_token : int,
                 hidden_dim : int,
                 base_encoder : tf.keras.Model,
                 team_vocab_size : int,
                 player_vocab_size : int,
                 season_vocab_size : int,
                 down_vocab_size : int):
        super(StratEncoder, self).__init__()
        
        self.SpeToken = tf.keras.layers.Embedding(input_dim = num_spec_token,
                                                   output_dim = hidden_dim)
        
        self.BaseEncoder = base_encoder
        self.SpeEncoder = MetaEmbedding(team_vocab_size=team_vocab_size,
                                        player_vocab_size=player_vocab_size,
                                        season_vocab_size=season_vocab_size,
                                        down_vocab_size=down_vocab_size,
                                        embedding_dim=hidden_dim)
        
        self.CrossAttention = SimpleTransformers(hidden_dim=hidden_dim,
                                                   output_dim=hidden_dim)
        
        self.Add = tf.keras.layers.Add()
        self.Conc = tf.keras.layers.Concatenate(axis = 1)
        
        self.Attention = SimpleTransformers(hidden_dim=hidden_dim,
                                                   output_dim=hidden_dim)
        
    def call(self, x):
        base_encoded = self.BaseEncoder(x)
        spe_encoded = self.SpeEncoder(x)
        
        cross_att = self.CrossAttention(q = spe_encoded, k = base_encoded, v = base_encoded)
        
        spec_token = self.SpeToken(x["spec_token"])
        play_embed = self.Conc([spec_token, cross_att])
        
        play_embed = self.Attention(q = play_embed, k = play_embed, v = play_embed)
        
        return play_embed
    
class StratFormer(tf.keras.Model):
    def __init__(self,
                 num_spec_token : int,
                 hidden_dim : int,
                 base_encoder : tf.keras.Model,
                 team_vocab_size : int,
                 player_vocab_size : int,
                 season_vocab_size : int,
                 down_vocab_size : int):
        super(StratFormer, self).__init__()
        
        self.Encoder = StratEncoder(num_spec_token=num_spec_token,
                                    hidden_dim=hidden_dim,
                                    base_encoder=base_encoder,
                                    team_vocab_size=team_vocab_size,
                                    player_vocab_size=player_vocab_size,
                                    season_vocab_size=season_vocab_size,
                                    down_vocab_size=down_vocab_size)
        
        self.Pred = tf.keras.layers.Dense(1, activation = "sigmoid")
        
    def call(self, x):
        encoded = self.Encoder(x)
        cls = encoded[:,0,:]
        pred = self.Pred(cls)
        return pred
        
        
        
        
        