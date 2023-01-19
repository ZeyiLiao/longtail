from torch.utils.data import Dataset, DataLoader


class Test2


class Test(Dataset):
	def __init__(self,ids):
		self.ids = ids

	def __len__(self):
		return len(self.ids)

	def __getitem__(self,index):
		return self.ids[index]


ids1= list(range(8))
ids2= list(range(9))

dataset1 = Test(ids1)
dataset2 = Test(ids2)
dataloader1 = DataLoader(dataset1,batch_size=4)
dataloader2 = DataLoader(dataset2,batch_size=4)
for batch in zip(dataloader1,dataloader2):
	print(batch)