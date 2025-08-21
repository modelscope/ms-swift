from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('swift/self-cognition', subset_name='default', split='train')