import numpy as np
import cv2
from PIL import Image
import math
import random
import string
import os
class IMAGE(str):
    def matrixgrayoutimage(self,filename,avHeight,avWidth,FullMatrix,i): #Load image converting to grayout and full matrix
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(filename+".jpg", gray )
        height, width, channels = img.shape
        column=0;
        if(height>avHeight):
            height = avHeight;
        if(width>avWidth):
            width = avWidth;
        im = Image.open(filename+".jpg") #Can be many different formats.
        pix = im.load()
        os.remove(filename+".jpg")
        for r in range(0,width):
            for c in range(0,height):
                pixnormalized =(float(pix[r,c])-float(128.0))/float(128.0);
                FullMatrix[i][column]=pixnormalized;
                column=column+1;
        return FullMatrix

    def dec_to_bin(self,x):
        return int(bin(x)[2:])

    def bin_vector(self,x):
        matrix = []
        for i in range(0,len(str(x))):
            num = str(x);
            matrix.append(int(num[i]));
        return matrix

class Arguments(object):
    def imagedirectory(self,path,filetype):
        #print path
        #print filetype
        list=[]
        avHeight=0;
        avWidth=0;
        for file in os.listdir(path):
            if file.endswith(filetype):
                list.append(path+"\\"+file);
                #print(os.path.join("/mydir",file))
                img = cv2.imread(path+"\\"+file)
                height, width, channels = img.shape
                avHeight+=height;
                avWidth+=width;
        avHeight/=(int)(len(list));
        avWidth/=(int)(len(list));
        return list,avHeight,avWidth
                
class Const(object): 
  def TotalImages(self,tot):
      self.total = tot;

  def get_TotalImages(self):
      return self.total
               


class NN:
  def __init__(self, NI, NH, NO,TI):
    # number of nodes in layers
    self.ni = NI + 1 # +1 for bias
    self.nh = NH
    self.no = NO
    self.ti = TI #Total number of images
    
    # initialize node-activations
    self.ai, self.ah, self.ao = [],[], []
    self.ai = [1.0]*self.ni
    self.ah = [1.0]*self.nh
    self.ao = [1.0]*self.no
    self.allwi=[] 
    self.allwo=[] 
    self.allci=[]
    self.allco=[]
    # create node weight matrices
    self.wi = makeMatrix (self.ni, self.nh)
    self.wo = makeMatrix (self.nh, self.no)
    for i in range(self.ti):
        # initialize node weights to random vals
        randomizeMatrix ( self.wi, -0.2, 0.2 )
        randomizeMatrix ( self.wo, -2.0, 2.0 )
        self.allwi.append(self.wi);
        self.allwo.append(self.wo);
    # create last change in weights matrices for momentum
        self.ci = makeMatrix (self.ni, self.nh)
        self.co = makeMatrix (self.nh, self.no)
        self.allci.append(self.ci);
        self.allco.append(self.co);
    
  def runNN (self, inputs,pattern):
    if len(inputs) != self.ni-1:
      print 'incorrect number of inputs'
    
    for i in range(self.ni-1):
      self.ai[i] = inputs[i]
      
    for j in range(self.nh):
      sum = 0.0
      for i in range(self.ni):
        sum +=( self.ai[i] * self.allwi[pattern][i][j] )
      self.ah[j] = sigmoid (sum)
    
    for k in range(self.no):
      sum = 0.0
      for j in range(self.nh):        
        sum +=( self.ah[j] * self.allwo[pattern][j][k] )
      self.ao[k] = sigmoid (sum)

    return self.ao
      
      
  
  def backPropagate (self, targets, N, M,image):
    # http://www.youtube.com/watch?v=aVId8KMsdUU&feature=BFa&list=LLldMCkmXl4j9_v0HeKdNcRA
    
    # calc output deltas
    # we want to find the instantaneous rate of change of ( error with respect to weight from node j to node k)
    # output_delta is defined as an attribute of each ouput node. It is not the final rate we need.
    # To get the final rate we must multiply the delta by the activation of the hidden layer node in question.
    # This multiplication is done according to the chain rule as we are taking the derivative of the activation function
    # of the ouput node.
    # dE/dw[j][k] = (t[k] - ao[k]) * s'( SUM( w[j][k]*ah[j] ) ) * ah[j]
    output_deltas = [0.0] * self.no
    for k in range(self.no):
      error = targets[k] - self.ao[k]
      output_deltas[k] =  error * dsigmoid(self.ao[k]) 
   
    # update output weights
    for j in range(self.nh):
      for k in range(self.no):
        # output_deltas[k] * |[j] is the full derivative of dError/dweight[j][k]
        change = output_deltas[k] * self.ah[j]
        self.allwo[image][j][k] += N*change + M*self.allco[image][j][k]
        ##self.wo[j][k] += N*change + M*self.co[j][k]
        self.allco[image][j][k] = change

    # calc hidden deltas
    hidden_deltas = [0.0] * self.nh
    for j in range(self.nh):
      error = 0.0
      for k in range(self.no):
        error += output_deltas[k] * self.allwo[image][j][k] #self.wo[j][k]
      hidden_deltas[j] = error * dsigmoid(self.ah[j])
    
    #update input weights
    for i in range (self.ni):
      for j in range (self.nh):
        change = hidden_deltas[j] * self.ai[i]
        #print 'activation',self.ai[i],'synapse',i,j,'change',change
        self.allwi[image][i][j] += N*change + M*self.allci[image][i][j]
        #self.wi[i][j] += N*change + M*self.ci[i][j]
        self.allci[image][i][j] = change
        
    # calc combined error
    # 1/2 for differential convenience & **2 for modulus
    #error = 0.0
    #for k in range(len(targets)):
      #error += 0.5 * (targets[k]-self.ao[k])**2
    
    mse = ((np.array(targets)-np.array(self.ao))**2).mean(axis=None);

    inp = open('outputsbackpropagation2.txt', 'a+')
    inp.write(str(targets)+str(mse)+'\n');
    inp.close()
    return mse
        
        
  def weights(self):
    print 'Input weights:'
    for i in range(self.ni):
      print self.wi[i]
    print
    print 'Output weights:'
    for j in range(self.nh):
      print self.wo[j]
    print ''
  
  def test(self, patterns):
    for p in range(len(patterns[0])):
      inputs = patterns[0][p]
      #print 'Inputs:', patterns[0][p], '-->', self.runNN(inputs,p), '\tTarget',
      print 'Inputs:', p, '-->', self.runNN(inputs,p), '\tTarget',
      patterns[1][p]
  
  def train (self, patterns, max_iterations = 1000, N=0.01, M=0.0001):
    error=0;
    for i in range(max_iterations):
      for p in range(len(patterns[0])):
        inputs = patterns[0][p]
        targets = patterns[1][p]
        self.runNN(inputs,p)
        error= self.backPropagate(targets, N, M,p)
      #if i % 50 == 0 and i!=0:
      print 'Combined error', error,'iteration',i
    self.test(patterns)
    inp = open('2weights_wi.txt', 'w')
    inp.write(str(self.allwi)+'\n');
    inp.close()
    inp = open('2weights_wo.txt', 'w')
    inp.write(str(self.allwo)+'\n');
    inp.close()

    

#def sigmoid (x):
  #return np.tanh(x)
  
def sigmoid(x):
  return 1 / (1 + np.exp(-x))


# the derivative of the sigmoid function in terms of output
# proof here: 
# http://www.math10.com/en/algebra/hyperbolic-functions/hyperbolic-functions.html
def dsigmoid2 (y):
  return 1 - y**2

def dsigmoid (y):
  return y*(1-y)

def makeMatrix ( I, J, fill=0):
  m = []
  for i in range(I):
    m.append([fill]*J)
  return m
  
def randomizeMatrix ( matrix, a, b):
  for i in range ( len (matrix) ):
    for j in range ( len (matrix[0]) ):
      matrix[i][j] = random.uniform(a,b)

def main ():
  
  new_image=IMAGE();
  image_list = Arguments();
  result,avHeight,avWidth = image_list.imagedirectory("F:\Google Drive\Proyecto Tesis Licenciatura\All Patterns\AgencyFB\\",".png");
  inputs=makeMatrix(len(result),avHeight*avWidth);
  total_lenght_outputs=len(str(new_image.dec_to_bin(len(result)))); 
  vectorOutput = makeMatrix(len(result),total_lenght_outputs); 
  for i in range(0,len(result)):
    inputs = new_image.matrixgrayoutimage(result[i],avHeight,avWidth,inputs,i);
    decnum=i+1;
    out =new_image.dec_to_bin(decnum);
    temp=total_lenght_outputs;
    tempidx = total_lenght_outputs-len(str(out));
    for j in range(0,len(str(out))):
        num = str(out);
        vectorOutput[i][tempidx+j]=int(num[j]);
    
  pat=inputs,vectorOutput
  l= len(inputs[0])
  print l
  
  hidden=((int)(math.ceil(math.sqrt(l*total_lenght_outputs))));
  myNN = NN(l, hidden,total_lenght_outputs,len(result))
  myNN.train(pat)

if __name__ == "__main__":
    main()
