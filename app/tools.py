import polars as pl
import numpy as np
import tensorflow as tf
import pandas as pd

class tokenizer:
    def __init__(self, 
                 moves_index : str,
                 play_index : str,
                 positions_index  : str,
                 scrimmage_index : str,
                 starts_index : str,
                 time_index : str,
                 window_size : int):
        self.window = window_size
        
        moves_index = pl.read_parquet(moves_index)
        self.moves_index = self.convert_index_to_dict(moves_index)
        
        play_index = pl.read_parquet(play_index)
        self.play_index= self.convert_index_to_dict(play_index)
        
        positions_index = pl.read_parquet(positions_index)
        self.positions_index = self.convert_index_to_dict(positions_index)
        
        scrimmage_index = pl.read_parquet(scrimmage_index)
        self.scrimmage_index = self.convert_index_to_dict(scrimmage_index)
        
        starts_index = pl.read_parquet(starts_index)
        self.starts_index = self.convert_index_to_dict(starts_index)
        
        time_index = pl.read_parquet(time_index)
        self.time_index = self.convert_index_to_dict(time_index)
        
        self.offdef_index = {0 : ["Def"],
                             1 : ["Off"]}
        
        self.index = {"input_ids" : self.moves_index,
                      "PlayType" : self.play_index,
                      "position_ids" : self.positions_index,
                      "scrim_ids" : self.scrimmage_index,
                      "start_ids" : self.starts_index,
                      "pos_ids" : self.time_index,
                      "OffDef" : self.offdef_index}
    
    def convert_index_to_dict(self, df : pl.DataFrame):
    
        ID_col = [v for v in df.columns if "ID" in v]
        assert len(ID_col) == 1
        new_id_name = ["ID"]

        val_cols = [v for v in df.columns if v not in ID_col+["Cat"]]
        new_val_name = ["Val_"+str(i) for i in range(1, len(val_cols)+1)]

        past_names = ID_col + val_cols
        new_names = new_id_name+new_val_name

        renaming = {past_names[i]: new_names[i] for i in range(len(new_names))}

        d = (df.
                drop("Cat").
                rename(renaming).
                select(new_names).
                to_dict(as_series=False))

        final_d = {d["ID"][i] : [d[k][i] for k in new_val_name] for i in range(len(d["ID"]))}

        return final_d
    
    def base_decode(self,
                    pad_element,
                    inputs : list,
                    index : dict,
                    first : bool):
        if first == True:
            return [index[v][0] if v in index.keys() else pad_element for v in inputs]
        else:
            return [index[v] if v in index.keys() else pad_element for v in inputs]
        
    def decode(self,
               inputs : list,
               type : str):
        if type in ["input_ids", "start_ids"]:
            padding = [-1000, -1000]
        elif type in ["scrim_ids", "pos_ids"]:
            padding = -1000
        else:
            padding = "[PAD]"
        
        if type in ["input_ids", "start_ids"]:
            return self.base_decode(padding, inputs, index = self.index[type], first=False)
        else:
            return self.base_decode(padding, inputs, index = self.index[type], first=True)

    def find_id_by_values(self, 
                          input_dict : dict, 
                          target_list : list):
        
        for key, values in input_dict.items():
            if set(target_list) == set(values):
                return key
        
    def base_encode(self,
                inputs : list,
                index : dict):
        return [self.find_id_by_values(index, [v]) for v in inputs]
    
    def encode(self,
               inputs : list,
               type : str):
        return self.base_encode(inputs, index = self.index[type])
    
    def decode_sequence(self,
                        input : dict):
        return {k : self.decode(v, k) if k not in ["side_ids", "token_type_ids", "labels", "attention_mask", "ids"] else v for k,v in input.items()}
    
    def encode_sequence(self,
                        input : dict):
        return {k : self.encode(v, k) if k not in ["side_ids", "token_type_ids", "labels", "attention_mask", "ids"] else v for k,v in input.items()}
    
    def truncate_to_time_t(self,
                           input : dict,
                           t : int):
        to_keep = [i < t for i in input["pos_ids"]]
        return {k: [v[i] for i in range(len(v)) if to_keep[i] == True] for k,v in input.items()}
    
    def resize_window(self,
                      input : dict,
                      pos_id):
        out = input.copy()
        out["attention_mask"] = [0 if out["pos_ids"][p] <pos_id else 1 for p in range(len(out["pos_ids"]))]
        return out
    
    def prepare_for_call(self, 
                         input : dict):
        resize_limit = max([v for v in np.array(input["pos_ids"]).flatten() if v != 51]) - self.window
        if resize_limit > 0:
            input = self.resize_window(input, resize_limit)
        
        done = {k : tf.constant(v) for k,v in input.items()}
        if len(done["pos_ids"].shape) == 1:
            done = {k : tf.expand_dims(v, axis=0) for k,v in input.items()}
        return done        
    
class generator:
    def __init__(self,
                 model,
                 tokenizer,
                 temp,
                 n_select):
        
        self.QBGPT = model
        self.tokenizer = tokenizer
        
        self.temperature = temp
        self.n_select = n_select
        
    def get_unique_lists(self,
                         l_of_ls : list):
        list_of_tuples = [tuple(inner_list) for inner_list in l_of_ls]
        
        # Create a set to eliminate duplicate
        unique_tuples = set(list_of_tuples)
        
        # Convert unique tuples back to lists
        unique_lists = [list(unique_tuple) for unique_tuple in unique_tuples]
        
        return unique_lists
                
    def cut(self, l, ref):
        splitted = []
        cutted = []
        for i in range(len(l)):
            if ref[i] == True:
                cutted.append(l[i])
            else:
                splitted.append(cutted)
                cutted = []
                cutted.append(l[i])     
            if i == len(l)-1:
                splitted.append(cutted) 
        return splitted
        
    def get_last_preds(self,
                       logits,
                       input : dict):
        
        to_keep = [i == max(input["pos_ids"]) for i in input["pos_ids"]]
        return np.array([logits[i] for i in range(len(logits)) if to_keep[i] == True])
        
    def get_logits(self, 
                   input : dict):
        x = self.tokenizer.prepare_for_call(input)
        return self.QBGPT(x)
    
    def convert_to_preds(self,
                         logits):
        preds = tf.squeeze(logits, axis=0)
        return preds
    
    def set_temperature(self, 
                        x):
        if x < 5:
            return self.temperature
        elif x < 10 and x >= 5:
            return self.temperature/2
        elif x <20 and x >= 10:
            return self.temperature/5
        else: 
            return 1.0
        
    def select_and_temp(self, 
                        tensor, 
                        n, 
                        temp):
        probas = tf.nn.softmax(tf.sort(tensor/temp, axis = -1)[:,:,-n:], axis = 2)
        indices = tf.argsort(tensor, axis = -1)[:,:,-n:]
        return probas, indices

    def draw_random(self,
                    probas):
        drawn = np.vstack([np.random.multinomial(1, p.numpy(), 1) for p in probas[0]])
        drawn = tf.expand_dims(drawn, axis = 0)
        return tf.cast(drawn, dtype="int32")
    
    def get_indices(self, 
                    drawn, 
                    ind):
        return tf.reduce_sum(drawn*ind, axis = 2)
    
    def process_logits(self,
                       logits,
                       temp,
                       n):
        probas, indices = self.select_and_temp(logits, n, temp)
        drawn = self.draw_random(probas)
        results = self.get_indices(drawn, indices)
        return results
    
    def generate(self,
                 input : dict):
        logits = self.get_logits(input)
        temperature_parameter = self.set_temperature(max(input["pos_ids"]))
        processed_logits = self.process_logits(logits, n=self.n_select, temp=temperature_parameter)
        preds = self.convert_to_preds(processed_logits)
        return self.get_last_preds(preds, input)
    
    def slice_inputs(self,
                     input : dict):
        flags = [True] + [input["pos_ids"][i+1] > input["pos_ids"][i] for i in range(len(input["pos_ids"])-1)]
        cutted_inputs = {k : self.cut(v, flags) for k,v in input.items()}
        return cutted_inputs
    
    def continue_by_token(self, 
                         arr,
                         token :str):
        if token == "input_ids":
            return arr
        if token == "pos_ids":
            insert = max(arr)+1
            return np.concatenate([arr, np.array([insert])])
        elif token == "token_type_ids":
            return  np.concatenate([arr, np.array([1])])
        else:
            return  np.concatenate([arr, [arr[-1]]])
            
        
    def append_prediction(self,
                           arr,
                           pred):
        return np.concatenate([arr, [pred]])
    
    def append_predictions(self,
                           d : dict,
                           preds):
        new = d.copy()
        new["input_ids"] = [self.append_prediction(new["input_ids"][i], preds[i]) for i in range(len(preds))]
        return new
                
    def merge_cuts(self, 
                   input : dict):
        return {k : np.concatenate(v) for k,v in input.items()}
        
    def update_inputs(self,
                      input,
                      preds):
        sliced = self.slice_inputs(input)
        appended = self.append_predictions(sliced, preds)
        continued = {k : [self.continue_by_token(e, k) for e in v] for k,v in appended.items()}
        merged = self.merge_cuts(continued)
        return merged
    
    def generate_sequence(self,
                          input,
                          t):
        new_input = input.copy()
        for i in range(t):
            generated = self.generate(new_input)
            new_input = self.update_inputs(new_input, generated)
        return new_input
    
    def convert_list(self,
                     d,
                     keep_original):
        new_df = d.copy()
        new_df["start_ids_x"] = [v[0] for v in new_df["start_ids"]]
        new_df["start_ids_y"] = [v[1] for v in new_df["start_ids"]]
        new_df["input_ids_x"] = [v[0] for v in new_df["input_ids"]]
        new_df["input_ids_y"] = [v[1] for v in new_df["input_ids"]]
        if keep_original == True:
            return new_df
        else:
            return {k : v for k,v in new_df.items() if k not in ["start_ids", "input_ids"]}
        
    def remove_pad(self,
                   seq):
        df = pd.DataFrame(seq)
        filtered = df[df["start_ids_x"] != -1000].reset_index(drop=True)
        filtered = df[df["input_ids_x"] != -1000].reset_index(drop=True)
        return filtered.to_dict(orient = "list")
    
    def _compute_true_sequence(self,
                          scrimmage_line,
                          start : list,
                          moves : list):
        scrimmage = np.array([scrimmage_line, 26.5])
        
        updated_moves = np.array([np.array(start) + np.array(v) for v in moves])
        
        appended = np.concatenate([np.expand_dims(start, axis = 0), updated_moves])
        
        final = appended + scrimmage
        return final
    
    def compute_true_sequence(self,
                              scrims,
                              starts,
                              moves):
        return self._compute_true_sequence(np.unique(scrims)[0], self.get_unique_lists(starts)[0], moves)
    
    def _resize_variable(self,
                         x, 
                         ref: str):
        
        if ref in ["pos_ids", "token_type_ids"]:
            return np.concatenate([[0], x])
        
        elif ref in ["input_ids", "start_ids"]:
            return np.vstack([self.get_unique_lists(x)[0], x])
        
        else:
            return np.concatenate([np.unique(x), x])
    
    def prepare_for_plot(self, 
                         seq):
        sequence = seq.copy()
        sequence = self.convert_list(sequence, keep_original = True)
        sequence = self.remove_pad(sequence)
        cutted = self.slice_inputs(sequence)
        moves_updated = [self.compute_true_sequence(cutted["scrim_ids"][i], cutted["start_ids"][i], cutted["input_ids"][i]) for i in range(len(cutted["input_ids"]))]
        cutted["input_ids"] = moves_updated
        cutted = {k : [self._resize_variable(e, k) if k != "input_ids" else e for e in v] for k,v in cutted.items()}
        cutted["ids"] = [[i for e in range(len(cutted["input_ids"][i]))] for i in range(len(cutted["input_ids"]))]
        merged = self.merge_cuts(cutted)
        converted = self.convert_list(merged, keep_original = False)
        structured = {k:v for k,v in converted.items() if k != "labels"}
        return structured

    def insert_ids(self,
                   input):
        
        cutted = self.slice_inputs(input)
        cutted["ids"] = [[i for e in range(len(cutted["input_ids"][i]))] for i in range(len(cutted["input_ids"]))]
        merged = self.merge_cuts(cutted)
        return merged
