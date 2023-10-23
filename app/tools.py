import polars as pl

class tokenizer:
    def __init__(self, 
                 moves_index : str,
                 play_index : str,
                 positions_index  : str,
                 scrimmage_index : str,
                 starts_index : str,
                 time_index : str):
        
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
        
        self.index = {"moves" : self.moves_index,
                      "plays" : self.play_index,
                      "positions" : self.positions_index,
                      "scrimmage" : self.scrimmage_index,
                      "starts" : self.starts_index,
                      "time" : self.time_index}
    
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
                    inputs : list,
                    index : dict):
        return [index[v] if v in index.keys() else "[PAD]" for v in inputs]
    
    def decode(self,
               inputs : list,
               type : str):
        return self.base_decode(inputs, index = self.index[type])
    
    def find_id_by_values(input_dict : dict, 
                          target_list : list):
        
        for key, values in input_dict.items():
            if set(target_list) == set(values):
                return key
        
    def base_encode(self,
                inputs : list,
                index : dict):
        return [self.find_id_by_values(index, v) for v in inputs]
    
    def encode(self,
               inputs : list,
               type : str):
        return self.base_encode(inputs, index = self.index[type])