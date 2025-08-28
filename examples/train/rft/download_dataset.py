#数据集下载
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('tastelikefeet/competition_math', subset_name='default', split='test', cache_dir="./data", trust_remote_code=True)
#您可按需配置 subset_name、split，参照“快速使用”示例代码
