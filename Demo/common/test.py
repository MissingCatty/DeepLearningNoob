import numpy as np

def forward(x,dropout_ratio=0.5, train_flg=True):
    drop_num=int(dropout_ratio*x.size)
    if train_flg:
        mask = np.zeros_like(x, dtype=bool)
        indices = np.random.choice(x.size, drop_num, replace=False)
        np.put(mask, indices, False)
        print(mask)
        return x*mask
    else:
        return x * (1.0 - dropout_ratio)


x=np.array(range(10))+1
x=x.reshape(2,5)
print(x)
print(forward(x))
