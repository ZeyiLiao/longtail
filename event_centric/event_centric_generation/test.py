def func1(x,*args,**kwargs):
	print(x)
	print(args)   
	print(kwargs)

func1(100,2,'a',('x','y'),'jack',age=20)

