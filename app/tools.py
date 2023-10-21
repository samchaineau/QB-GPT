import polars as pl

class tokenizer:
    def __init__(self, 
                 moves_index : str,
                 play_index : str,
                 positions_index  : str,
                 scrimmage_index : str,
                 starts_index : str,
                 time_index : str):
        
        self.moves_index = pl.read_parquet(moves_index)
        self.moves_index = self.convert_index_to_dict(self.moves_index)
        
        self.play_index = pl.read_parquet(play_index)
        self.play_index= self.convert_index_to_dict(self.play_index)
        
        self.positions_index = pl.read_parquet(positions_index)
        self.positions_index = self.convert_index_to_dict(self.positions_index)
        
        self.scrimmage_index = pl.read_parquet(scrimmage_index)
        self.scrimmage_index = self.convert_index_to_dict(self.scrimmage_index)
        
        self.starts_index = pl.read_parquet(starts_index)
        self.starts_index = self.convert_index_to_dict(self.starts_index)
        
        self.time_index = pl.read_parquet(time_index)
        self.time_index = self.convert_index_to_dict(self.time_index)
        
    @staticmethod
    def convert_index_to_dict(df : pl.DataFrame):
    
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
    
    @staticmethod
    def _decode(inputs : list,
                index : dict):
        return [index[v] for v in inputs]
    
    def decode(inputs : list,
                index : dict):
        return _decode(inputs = inputs,
                       index = index)
    
    @staticmethod
    def find_id_by_values(input_dict : dict, 
                          target_list : list):
        
        for key, values in input_dict.items():
            if set(target_list) == set(values):
                return key
        
    def _encode(self,
                inputs : list,
                index : dict):
        return [self.find_id_by_values(index, v) for v in inputs]
    
    
    
    