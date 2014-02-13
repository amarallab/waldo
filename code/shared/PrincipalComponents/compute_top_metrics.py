
from in_output_routines import get_data_from_file

def compute_overlap(top_m,prc):
    
    sum=0.
    for v in top_m:
        sum+=prc[v[1]]*prc[v[1]]
    return sum


x=get_data_from_file('prcs.txt')

top_metrics=[]

print len(x)
for v in x[:5]:
    top_metrics_v=[]
    for k,a in enumerate(v):
        top_metrics_v.append((a*a, k))
    top_metrics_v=sorted(top_metrics_v, reverse=True)
    top_metrics.append(top_metrics_v)


#print top_metrics


f=open('2012-10-27_metric_index.txt', 'r')
names=[]
for l in f.readlines():
    names.append(l.split()[0])





for k,v in enumerate(top_metrics):

    top_m=v[:20]
    for a in top_m:
        print names[a[1]], a[0], '*********'
        
    for j in xrange(5):
        ov=compute_overlap(top_m,x[j])
        print k+1,j+1,ov
