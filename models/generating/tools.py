import polars as pl
import numpy as np
from collections import defaultdict


class QBGPT_steward:
    def __init__(self, 
                 path_index_moves : str,
                 path_index_plays : str,
                 path_index_positions: str,
                 path_index_scrimmage: str,
                 path_index_starts : str,
                 path_most_probable_starts : str,
                 path_most_probable_pos):
        
        self.moves_index = pl.read_parquet(path_index_moves)
        self.plays_index = pl.read_parquet(path_index_plays)
        self.positions_index = pl.read_parquet(path_index_positions)
        self.scrimmage_index = pl.read_parquet(path_index_scrimmage)
        self.starts_index = pl.read_parquet(path_index_starts)
        
        
        self.most_probable_starts_per_pos = pl.read_parquet(path_most_probable_starts)
        self.most_probable_pos_per_team = pl.read_parquet(path_most_probable_pos)
        
    @staticmethod
    def merge_dicts(dicts_list):
        merged_dict = defaultdict(list)
        
        for dictionary in dicts_list:
            for key, value in dictionary.items():
                merged_dict[key].append(value)
            
        return dict(merged_dict)
        
        
    def generate_random_play(self):
        sampled_play = self.plays_index.select("PlayType").sample(n=1).to_series().to_list()[0]
        return sampled_play
    
    def generate_random_scrimmage(self):
        sampled_scrimmage = self.scrimmage_index.select("line_scrimmage").sample(n=1).to_series().to_list()[0]
        return sampled_scrimmage
    
    def generate_random_position(self, team_id: int):
        sampled_position = self.most_probable_pos_per_team.filter(pl.col("OffDef_ID") == team_id).select("position_ID").to_series().to_list()[0]
        return sampled_position
    
    def generate_random_starts(self, position : str):
        position_to_pos_id = self.positions_index.filter(pl.col("position") == position).select("position_ID").to_series().to_list()[0]
        sampled_start_ID = self.plays_index.filter(pl.col("position_ID") == position_to_pos_id).select("Start_ID").sample(n=1).to_series().to_list()[0]
        sampled_start = self.starts_index.filter(pl.col("Start_ID") == sampled_start_ID).sample(n=1).select("Start_x", "Start_y").to_series().to_list()
        return sampled_start
    
    
    def generate_single_input(self,
                              team_id: int,
                              position : int = None,
                              scrimmage_line : int = None,
                              starts : list = None):
        if position is None:
            position = self.generate_random_position(team_id)
        
        if scrimmage_line is None:
            scrimmage_line = self.generate_random_scrimmage()
        
        if starts is None:
            starts = self.generate_random_starts(position)
        
        return {"team" : team_id,
                "position" : position,
                "scrimmage" : scrimmage_line,
                "starts" : starts}
        
    def generate_team_input(self,
                            team_id : int,
                            position_list : int = [None],
                            scrimmage_lint : int = None,
                            starts : list = [None]):
        
    def generate_output(self, outputs):
         

convert_input 

convert_output 

plot_traj