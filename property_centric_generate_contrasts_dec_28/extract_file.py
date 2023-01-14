import json
import glob
import csv
import random
random.seed(42)

def verbal_into_stm(actions,obj,property, if_increase = False):
	direction = "increase" if if_increase else "decrease"
	stms = []
	for action in actions:
		stm = f"If {action}, the {property} of {obj} will {direction}"
		stms.append(stm)
	return stms

def check_attr(dict):
	count = 0
	for x in dict:
		if dict[x].get("rewritten_action",None) == None:
			continue
		count += 1
	return count


def check_length(all_actions,obj1,obj2):
	if check_attr(all_actions[obj1]['increase'])>= 3 and check_attr(all_actions[obj1]['decrease']) >= 3:
		if check_attr(all_actions[obj2]['increase'])>= 3 and check_attr(all_actions[obj2]['decrease']) >= 3:
			return True

	return False

def samples(action1):
	increase1 = action1['increase']
	increase1_keys = increase1.keys()
	increase1_keys = [increase1[x].get("rewritten_action",None) for x in increase1_keys]
	increase1_keys = list(filter(lambda i: i != None,increase1_keys))
	increase1_sample = random.sample(increase1_keys,3)


	decrease1 = action1['decrease']
	decrease1_keys = decrease1.keys()
	decrease1_keys = [decrease1[x].get("rewritten_action",None) for x in decrease1_keys]
	decrease1_keys = list(filter(lambda i: i != None,decrease1_keys))
	decrease1_sample = random.sample(decrease1_keys,3)

	return increase1_sample,decrease1_sample


root_path = "../property_centric"
root_paths = glob.glob(f"{root_path}/*")
exclude_paths = glob.glob(f"{root_path}/*.txt")
root_paths = list(set(root_paths) - set(exclude_paths))
for path in root_paths:
	property = path.split('/')[-1]

	action_file = glob.glob(f"{path}/updated_actions*")[0]
	relevant_file = glob.glob(f"{path}/filtered_object*")[0]
	filtered_object_file = glob.glob(f"{path}/objects_extra*")[0]


	filtered_objects =  [x.strip() for x in open(filtered_object_file).readlines()]
	relevant_objects = []
	relevant_objects_actions = []

	total_count = 0

	with open(action_file) as f:
		all_actions = json.load(f)


	with open(relevant_file) as f:
		reader = csv.reader(f)
		for line in reader:
			obj1, obj2 = line[0],line[1]
			if obj1 in filtered_objects and obj2 in filtered_objects and check_length(all_actions,obj1,obj2):
				relevant_objects.append((obj1,obj2))
				total_count += 1
			if total_count >= 400:
				break

	
	all_stmts = []
	for relevant_object in relevant_objects:
		obj1, obj2 = relevant_object[0],relevant_object[1]

		action1 = all_actions[obj1]
		action2 = all_actions[obj2]

		increase1,decrease1 = samples(action1)
		increase2,decrease2 = samples(action2)
		tmp = {}
		tmp['obj1'] = obj1
		tmp['obj2'] = obj2
		tmp['obj1_increase'] = increase1
		tmp['obj1_decrease'] = decrease1
		tmp['obj2_increase'] = increase2
		tmp['obj2_decrease'] = decrease2
		stmts = []
		stmts.append(verbal_into_stm(increase1,obj2,property))
		stmts.append(verbal_into_stm(decrease1,obj2,property,if_increase=True))
		stmts.append(verbal_into_stm(increase2,obj1,property))
		stmts.append(verbal_into_stm(decrease2,obj1,property,if_increase=True))
		tmp['stmts'] = stmts
		all_stmts.append(tmp)

	with open(f"{path}/check_conflicts.json",'w') as f:
		json.dump(all_stmts,f)
		

