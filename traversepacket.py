import pandas as pd
import math
path="data\\compiled_node_dataset.csv"
table=pd.read_csv(path)
print(table,"\n")             #prints actual table

temp_table=table[['Battery','Internal','Range','Class']]
print("Trimmed dataset is:\n",temp_table)        #prints altered table

#routing algorithm
def creatematrix():   #this will generate the distance matrix for the evaluasstion 
    dist_matrix=list()
    
    for i in range(len(temp_table)):
        col=[]
        for j in range(len(temp_table)):
            if(temp_table.iloc[i,0]==temp_table.iloc[j,0]):
                col.append(0)
            else:
                val1=(temp_table.iloc[i,-2]-temp_table.iloc[j,-2])**2
                val2=(temp_table.iloc[i,-1]-temp_table.iloc[j,-1])**2
                col.append(math.sqrt(val1+val2))
        dist_matrix.append(col)
    return dist_matrix   

def traverse(matrixx,sender,receiver):
    neighbor_nodes=[]
    neighbor_nodeclass=[]
    node_dict=dict()
    node_range=temp_table.iloc[sender,-4] #will fetch the range of sender from the temp_tables
    if(sender == receiver):
        print("Packet received\n\n")
        return(0)
    for i in matrixx[sender]:
        if(matrixx[sender][i]<=node_range):
            neighbor_nodes.append(i)
        
    neighbor_nodeclass=KNN(neighbor_nodes)  # KNN represents the function module created for the classification  
    node_dict = {neighbor_nodes[i]: neighbor_nodeclass[i][5] for i in range(len(neighbor_nodes))}     
    node=max(node_dict,key = node_dict.get)
    print("Node {}".format(node))
    return traverse(matrixx,node,receiver)
    
def main():
    dist_matrix=[]
    sender=int(input("Please enter the Sender node\n\n"))
    receiver=int(input("Please enter the Receiver node\n\n"))
    dist_matrix=creatematrix()
    traverse(dist_matrix,sender,receiver)

main()
