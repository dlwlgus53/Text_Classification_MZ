import pandas as pd
data = pd.read_csv("./data/train.txt", sep = '\t', header = None)
import pdb; pdb.set_trace()
nums = len(data[0].unique())

key_value = pd.DataFrame({'key' : data[0].unique(), 'value' : range(0,nums) })
key_value.to_csv("./data/class_embedding.csv",index = False )
