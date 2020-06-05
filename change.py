f=open('ship_val_final_new2.csv').readlines()
for l in range(len(f)):
	f[l]='train_v3/'+f[l][9:]
	#f[l]='train_v2/'+f[l]
	
g=open('ship_val_final_new2.csv','w')
g.write("".join(f))