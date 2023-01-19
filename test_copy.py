import re

text = 'This string11 is an example for match string2'

result2 = re.findall(r'This (\w*) is an example for match (.*$)',text)

pattern = re.compile(r"This(.*?) is an example for match (.*$)")

pattern2 = re.compile(r"This \w*")

print(result2)
print(pattern.findall(text))