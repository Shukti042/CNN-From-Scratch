from keras.datasets import mnist,cifar10
import numpy as np
from sklearn.metrics import classification_report

no_of_classes=0
alpha=0.001
def preprocess_mnist():
    global no_of_classes,alpha
    no_of_classes=10
    alpha=0.001
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x=np.reshape(train_x,(train_x.shape[0],train_x.shape[1],train_x.shape[2],1))
    train_y=np.reshape(train_y,(train_y.shape[0],1))
    test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1],test_x.shape[2],1))
    test_y=np.reshape(test_y,(test_y.shape[0],1))
    return train_x, train_y, test_x, test_y
def preprocess_cifar():
    global no_of_classes,alpha
    alpha=0.001
    no_of_classes=10
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    return train_x, train_y, test_x, test_y

def cost(y_hat,y):
    temp=np.log(y_hat)
    temp=np.multiply(temp,y)
    temp=temp*-1
    temp=np.sum(temp,axis=1)
    return np.sum(temp)/np.size(temp)

class Softmax:
    def __init__(self):
        pass
    def forward(self,x):
        x=x-np.amax(x,axis=0)
        x=np.exp(x)
        x=x/np.sum(x,axis=0)
        return(x)
    def backward(self,y,y_hat):
        return (y_hat-y).T
class Flatten:
    original_shape=None
    def __init__(self):
        pass
    def forward(self,x):
        self.original_shape=x.shape
        dim=np.prod(x.shape)
        dim=int(dim/x.shape[0])
        r=np.empty((x.shape[0],dim))
        for index,item in enumerate(x):
            temp=item.flatten()
            r[index,:]=temp
        return r.T
    def backward(self,x):
        temp=x.T
        temp=temp.reshape(self.original_shape)
        return temp
class Conv:
    W=None
    b=None
    stride=None
    pad=None
    A_prev=None
    def __init__(self,s,p):
        self.stride=s
        self.pad=p
    def zero_pad(self,x):
        xp=np.pad(x,((0,0),(self.pad,self.pad),(self.pad,self.pad),(0,0)),'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
        return xp
    def conv_single_step(self,a,w,b):
        s = np.multiply(a,w)
        z=np.sum(s,axis=(1,2,3))
        b = np.squeeze(b)
        z = z + b
        return z
    def forward(self,A_prev):
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (f, f, n_C_prev, n_C) = self.W.shape
        n_H = int((n_H_prev + 2*self.pad - f)/self.stride) + 1
        n_W = int((n_W_prev + 2*self.pad - f)/self.stride) + 1
        Z = np.zeros((m, n_H, n_W, n_C))
        A_prev_pad = self.zero_pad(A_prev)       
        for h in range(n_H):        
            vert_start = self.stride * h 
            vert_end = vert_start  + f
            for w in range(n_W): 
                horiz_start = self.stride * w
                horiz_end = horiz_start + f
                for c in range(n_C):  
                    a_slice_prev = A_prev_pad[:,vert_start:vert_end,horiz_start:horiz_end,:]
                    weights = self.W[:, :, :, c]
                    biases  = self.b[:, :, :, c]
                    Z[:, h, w, c] = self.conv_single_step(a_slice_prev, weights, biases)
        self.A_prev = A_prev
        return Z
    def initialize(self,f,c_i,c_o,prev_input_size):
        self.W=np.random.randn(f,f,c_i,c_o)*np.sqrt(2/prev_input_size)
        self.b=np.zeros((1,1,1,c_o))
    def backward(self,dZ,alpha=0.001):
        (m, n_H_prev, n_W_prev, n_C_prev) = self.A_prev.shape
        (f, f, n_C_prev, n_C) = self.W.shape
        (m, n_H, n_W, n_C) = dZ.shape
        dA_prev = np.zeros((self.A_prev.shape))                        
        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)
        A_prev_pad = self.zero_pad(self.A_prev)
        dA_prev_pad = self.zero_pad(dA_prev)
        for h in range(n_H):                   
            for w in range(n_W):               
                for c in range(n_C):           
                    vert_start = self.stride * h 
                    vert_end = vert_start + f
                    horiz_start = self.stride * w
                    horiz_end = horiz_start + f
                    a_slice = A_prev_pad[:,vert_start:vert_end,horiz_start:horiz_end,:]
                    dA_prev_pad[:,vert_start:vert_end,horiz_start:horiz_end,:] += self.W[np.newaxis,:,:,:,c] * dZ[:, h:h+1, w:w+1,np.newaxis,c]
                    dW[:,:,:,c] += np.sum(a_slice * dZ[:, h:h+1, w:w+1,np.newaxis,c],axis=0)
                    db[:,:,:,c] += np.sum(dZ[:, h, w, c])
        if self.pad!=0:
            dA_prev=dA_prev_pad[:,self.pad:-self.pad,self.pad:-self.pad,:]
        else:
            dA_prev = dA_prev_pad
        self.W=self.W-(dW*alpha)
        self.b=self.b-(db*alpha)
        return dA_prev

class Relu:
    cache=None
    def __init__(self):
        pass
    def forward(self,x):
        self.cache=x
        return np.maximum(x,0)
    def backward(self,dA):
        self.cache[self.cache<=0] = 0
        self.cache[self.cache>0] = 1
        return self.cache*dA
class Pool:
    f=None
    stride=None
    A_prev=None
    def __init__(self,f,s):
        self.f=f
        self.stride=s
    def forward(self,A_prev):
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        n_H = int(1 + (n_H_prev - self.f) / self.stride)
        n_W = int(1 + (n_W_prev - self.f) / self.stride)
        n_C = n_C_prev
        A = np.zeros((m, n_H, n_W, n_C))                                     
        for h in range(n_H):                     
            vert_start = self.stride * h 
            vert_end = vert_start + self.f
            for w in range(n_W):                 
                horiz_start = self.stride * w
                horiz_end = horiz_start + self.f        
                a_slice_prev = A_prev[:,vert_start:vert_end,horiz_start:horiz_end,:]
                A[:, h, w, :] = np.max(a_slice_prev,axis=(1,2))
        self.A_prev = A_prev
        return A
    def backward(self,dA):
        m, n_H_prev, n_W_prev, n_C_prev = self.A_prev.shape
        m, n_H, n_W, n_C = dA.shape
        dA_prev = np.zeros(self.A_prev.shape)

        for h in range(n_H):                   
            for w in range(n_W):              
                for c in range(n_C):           
                    vert_start  = h * self.stride
                    vert_end    = h * self.stride + self.f
                    horiz_start = w * self.stride
                    horiz_end   = w * self.stride + self.f
                    
                    a_prev_slice = self.A_prev[:,vert_start:vert_end, horiz_start:horiz_end, c ]
                    mask = (a_prev_slice == np.amax(a_prev_slice,axis=(1,2)).reshape(a_prev_slice.shape[0],1,1))
                    dA_prev[:,vert_start:vert_end, horiz_start:horiz_end,c] += (mask * dA[:, h:h+1, w:w+1, c])
        
        
        return dA_prev
class FC:
    W=None
    b=None
    A_prev=None
    def __init__(self):
        pass
    def forward(self,A):
        Z = np.dot(self.W,A) + self.b
        self.A_prev = A
        return Z
    def initialize(self,o_dim,i_dim):
        self.W=np.random.randn(o_dim,i_dim)*np.sqrt(2/i_dim)
        self.b=np.zeros((o_dim,1))

    def backward(self,dZ,alpha=0.001):
        m = self.A_prev.shape[1]
        dW = (1/m) * np.dot(dZ, self.A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T,dZ)
        self.W=self.W-(dW*alpha)
        self.b=self.b-(db*alpha)
        return dA_prev

def main():
    train_x, train_y, test_x, test_y=preprocess_mnist()
    np.random.seed(1)
    train_x=train_x/255
    test_x=test_x/255
    x=train_x[:2,:,:,:]
    np.random.seed(1)
    layers=[]
    gotFlatten=False
    gotFC=False
    inputfile=open("input",'r')
    lines=inputfile.readlines()
    z=x.copy()
    for line in lines:
        items=line.split()
        if items[0]=='Conv':
            n_o_c=int(items[1])
            f=int(items[2])
            s=int(items[3])
            p=int(items[4])
            conv=Conv(s,p)
            conv.initialize(f,z.shape[-1],n_o_c,np.size(z[0]))
            z=conv.forward(z)
            layers.append(conv)
            continue
        if items[0]=='ReLU':
            relu=Relu()
            z=relu.forward(z)
            layers.append(relu)
            continue
        if items[0]=='Pool':
            f=int(items[1])
            s=int(items[2])
            pool=Pool(f,s)
            z=pool.forward(z)
            layers.append(pool)
            continue
        if items[0]=='Flatten':
            flat=Flatten()
            z=flat.forward(z)
            layers.append(flat)
            gotFlatten=True
            continue
        if items[0]=='FC':
            if not gotFlatten and not gotFC:
                flat=Flatten()
                z=flat.forward(z)
                layers.append(flat)
                gotFlatten=True
                gotFC=True
            output_dim=int(items[1])
            fc=FC()
            fc.initialize(output_dim,z.shape[0])
            z=fc.forward(z)
            layers.append(fc)
            continue
        if items[0]=='Softmax':
            soft=Softmax()
            z=soft.forward(z)
            layers.append(soft)
            continue
    inputfile.close

    batchsize=32
    totaldata=train_x.shape[0]
    validationdata=int(test_x.shape[0]/2)
    global alpha
    for i in range(5):
        idx=0
        flag=False
        if(i%2!=0):
            alpha=alpha/2
        while True:
            x=None
            yt=None
            if idx+batchsize>=totaldata:
                flag=True
                x=train_x[idx:,:,:,:]
                yt=train_y[idx:,:].flatten()
            else:
                x=train_x[idx:idx+batchsize,:,:,:]
                yt=train_y[idx:idx+batchsize,:].flatten()
            idx+=batchsize
            z=x.copy()

            for layer in layers:
                z=layer.forward(z)
            z=z.T
            
            y=np.zeros((yt.size, no_of_classes))
            y[np.arange(yt.size),yt] = 1
            # print(cost(z,y))
            # yp=z.argmax(axis=1)
            # print(classification_report(yt,yp))
            dA=None
            for layer in reversed(layers):
                if isinstance(layer,Softmax):
                    dA=layer.backward(y,z)
                elif isinstance(layer,Conv) or isinstance(layer,FC):
                    dA=layer.backward(dA,alpha)
                else:
                    dA=layer.backward(dA)
            if flag:
                break
        print("------------Epoch ",i+1,"-------------")
        print()
        print("--------------Training Set--------------")
        z=train_x.copy()
        yt=train_y.copy().flatten()
        for layer in layers:
            z=layer.forward(z)
        z=z.T
        y=np.zeros((yt.size, no_of_classes))
        y[np.arange(yt.size),yt] = 1
        print("Loss : ",cost(z,y))
        yp=z.argmax(axis=1)
        print(classification_report(yt,yp))
        print()
        print("--------------Validation Set--------------")
        z=test_x[:validationdata,:,:,:].copy()
        yt=test_y[:validationdata,:].copy().flatten()
        for layer in layers:
            z=layer.forward(z)
        z=z.T
        y=np.zeros((yt.size, no_of_classes))
        y[np.arange(yt.size),yt] = 1
        print("Loss : ",cost(z,y))
        yp=z.argmax(axis=1)
        print(classification_report(yt,yp))
        print()
    print("--------------Test Set--------------")
    z=test_x[validationdata:,:,:,:].copy()
    yt=test_y[validationdata:,:].copy().flatten()
    for layer in layers:
        z=layer.forward(z)
    z=z.T
    y=np.zeros((yt.size, no_of_classes))
    y[np.arange(yt.size),yt] = 1
    print("Loss : ",cost(z,y))
    yp=z.argmax(axis=1)
    print(classification_report(yt,yp))
    print()
    

    
if __name__ == '__main__':
    main()