from PIL import Image
import cv2
import math
import numpy as np
import os
#import decimal
class IMAGE(str):
    #def matrixgrayoutimage(self): #Load image converting to grayout and full matrix
    #    matrix=[]
    #    img = cv2.imread("C:\\Users\\Jacob\\Downloads\\    PATTERNS2\\3.bmp")
    #    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #    cv2.imwrite( "C:\\Users\\Jacob\\Downloads\\PATTERNS2\\3.jpg", gray )
    #    height, width, channels = img.shape
    #    total=0;
    #    im = Image.open("C:\\Users\\Jacob\\Downloads\\PATTERNS2\\3.jpg") #Can be many different formats.
    #    pix = im.load()
    #    for r in range(0,width):
    #        for c in range(0,height):
    #            matrix.append((float(pix[r,c])-float(128.0))/float(128.0));
    #    return matrix

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

class ReadWeights():
    def weightsList(self):
        arr = []
        #inp = open ("F:\\Google Drive\\Proyecto Tesis Licenciatura\\All Patterns\\AgencyFB\\weights.txt","r")
        inp = open ("C:\\Users\\Jacob\\Pictures\\Google Photos Backup\\weights.txt","r")
        #read line into array 
        c=0;
        for line in inp.readlines():
            # loop over the elemets, split by whitespace
            list = []
            for i in line.split():
                temp="";
                for j in range(len(i)):
                    if (i[j].__str__()!="[" and i[j].__str__()!="]" and i[j].__str__()!="," and i[j].__str__()!="'"):
                        temp=temp+i[j].__str__()
                list.append(float(temp))
                # convert to integer and append to the list
            arr.append(list)
            c=c+1;
            print "Weights loaded from image #",c;
            
        return arr;    

    def outputList(self):
        #inp = open ("F:\\Google Drive\\Proyecto Tesis Licenciatura\\All Patterns\\AgencyFB\\arquitecture_nn.txt","r")
        inp = open ("C:\\Users\\Jacob\\Pictures\\Google Photos Backup\\arquitecture_nn.txt","r")
        
        #read line into array 
        data = []
        for line in inp.readlines():
            # loop over the elemets, split by whitespace
            for i in line.split():
                data.append(i)
        return data; 
    def trained_mse_list(self):
        inp = open ("F:\\Google Drive\\Proyecto Tesis Licenciatura\\All Patterns\\AgencyFB\\mse_trained.txt","r")
        #read line into array 
        data = []
        for line in inp.readlines():
            # loop over the elemets, split by whitespace
            for i in line.split():
                data.append(i)
        return data; 
class Const(object):
  def TotalImages(self,tot):
      self.total = tot;
  
  def get_TotalImages(self):
      return self.total;

  def avHeight(self,Height):
      self.avHeight =Height;

  def get_avHeight(self):
      return self.avHeight;

  def avWidth(self,Width):
      self.avWidth = Width;

  def get_avWidth(self):
      return self.avWidth;

  def numInput(self,numInput):
      self.numInput = numInput;

  def get_numInput(self):
      return self.numInput;

  def numHidden(self,numHidden):
      self.numHidden = numHidden;

  def get_numHidden(self):
      return self.numHidden;
      
  def numOutput(self,numOutput):
      self.numOutput = numOutput;

  def get_numOutput(self):
      return self.numOutput;

  def output(self,output):
      self.output = output;

  def get_output(self):
      return self.output;

  def D(self,numInput,numHidden,numOutput):
      self.numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;

  def get_D(self):
      return self.numWeights;
  
  def MatrixInput(self,matrix):
      self.matrix = matrix;
  
  def get_MatrixInput(self):
      return self.matrix;

  def output(self,output):
      self.output = output;

  def get_output(self):
      return self.output;

  def MakeMatrix ( self,I, J, fill=0):
    m = []
    for i in range(I):
      m.append([fill]*J)
    return m


class NNRecognize():
    def MakeMatrix ( self,I, J, fill=0):
        m = []
        for i in range(I):
            m.append([fill]*J)
        return m

    def GetColumnMatrix(self,Foods,rows,idx_j):
        for i in range (rows):
            ColumVector = []
            ColumVector.append([Foods[i][idx_j]]);
        return ColumVector

    def GetVectorElements(self,vector,a,b):
        rows = 1;
        columns = b+1-a;
        x=[]   
        x= NNRecognize.MakeMatrix(self,rows,columns);
        for i in range ( len (x) ):
            for j in range ( len (x[0]) ):
                x[i][j] = vector[a+j-1];
            #a=a+1;
        return x  
     
    #Sigmoid function
    def Sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
        #decimal.setcontext(decimal.Context(prec=40))
        #return (float)(decimal.Decimal(1) / (decimal.Decimal(1) + decimal.Decimal(-x).exp()))
    
    
    #Hyperbolic tangent function
    def sigmoid (self,x):
        return math.tanh(x)

    def ReadVector(self,vector):
        rows=len(vector);
        columns = len(vector[0]);
        m=NNRecognize.MakeMatrix(self,rows,columns)  
        for k in range(rows):
            for j in range(columns):
                m[k][j]=NNRecognize.Sigmoid(self,vector[k][j]); 
        return m
    def CalculateOutput(self,Foods,image):
        rows=len(Foods);
        columns = len(Foods[0]);
        x=[]
        outputList=[]
        mseList=[]
        matrixIn = C.get_MatrixInput();
        matrixOut = C.get_output();
        maxValue=1;
        MatchedHigh="?";
        MatchedLow="?";
        for ni in range(rows):
            x= Foods[ni]
            trout = matrixOut[ni];
            self.trin = matrixIn[0];
            iw = NNRecognize.GetVectorElements(self,x,1,C.get_numInput()*C.get_numHidden()); 
            b1 = NNRecognize.GetVectorElements(self,x,C.get_numInput()*C.get_numHidden()+1,C.get_numInput()*C.get_numHidden()+C.get_numHidden()); 
            lw = NNRecognize.GetVectorElements(self,x,C.get_numInput()*C.get_numHidden()+C.get_numHidden()+1,C.get_numInput()*C.get_numHidden()+C.get_numHidden()+C.get_numHidden()*C.get_numOutput());
            b2 = NNRecognize.GetVectorElements(self,x,C.get_numInput()*C.get_numHidden()+C.get_numHidden()+C.get_numHidden()*C.get_numOutput()+1,C.get_numInput()*C.get_numHidden()+C.get_numHidden()+C.get_numHidden()*C.get_numOutput()+C.get_numOutput());
            
            ihSums = C.MakeMatrix(1,C.get_numHidden());
            
            c=0;
            for i in range(0,C.get_numHidden()):
                for j in range(0,C.get_numInput()):    
                    ihSums[0][i] += self.trin[j]*iw[0][c]; 
                    c=c+1;
                        
            for k in range(0,C.get_numHidden()):
                ihSums[0][k] +=  b1[0][k]; 
                          
            ihOutputs = NNRecognize.ReadVector(self,ihSums);
            hoSums = C.MakeMatrix(1,C.get_numOutput());
            
            
            c=0;
            for l in range(0,C.get_numOutput()):
                for m in range(0,C.get_numHidden()):    
                    hoSums[0][l] += ihOutputs[0][m]*lw[0][c];
                    c=c+1;
                  
            for n in range(0,C.get_numOutput()):
                hoSums[0][n] += b2[0][n];
                    
            outputs = NNRecognize.ReadVector(self,hoSums);         
            mse = ((np.array([trout])-np.array(outputs))**2).mean(axis=None);
            mseList.append(mse);
            outputList.append(outputs);
            #if(float(msetrained[ni])==float('%.17f'%mse) or (float(msetrained[ni])>=float('%.17f'%mse)*0.001+float('%.17f'%mse))):
            #    print "La imagen reconocida es:",trout
            if(float(mse)<maxValue): #and  float(mse) >= float(msetrained[ni])):
                MatchedLow=MatchedHigh;
                OutputValueHigh=maxValue;
                maxValue=float('%.17f'%mse);
                MatchedHigh=trout;
                OutputValueHight = maxValue;

            inp = open('C:\\Users\\Jacob\\Pictures\\Google Photos Backup\\outputsR.txt', 'a+')
            #inp.write(str('[Output Desired]')+str('[Final output]')+str('[MSE]')+'\n');
            inp.write(str(trout)+str(outputs)+str(mse)+'\n');
            inp.close()
        print("Matched High:"+str(MatchedHigh)+'\n');
        #print("Matched Low:"+str(MatchedLow)+'\n');
        inp = open('C:\\Users\\Jacob\\Pictures\\Google Photos Backup\\recognitionResults.txt', 'a+')
        #inp.write(str('[Output Desired]')+str('[Final output]')+str('[MSE]')+'\n');
        inp.write(str(MatchedHigh)+'\n');
        #inp.write("Matched Low:"+str(MatchedLow)+'\n');
        inp.close()
        inp = open('C:\\Users\\Jacob\\Pictures\\Google Photos Backup\\mseRecognition.txt', 'a+')
        inp.write(str(mseList[image])+'\n');
        inp.close()
        return mse;
    
    def dec_to_bin(self,x):
        return int(bin(x)[2:])

    def bin_vector(self,x):
        matrix = []
        for i in range(0,len(str(x))):
            num = str(x);
            matrix.append(int(num[i]));
        return matrix
img=[]
wl=[]
C = Const();
new_image = IMAGE();
weightsList = ReadWeights();
evaluate= NNRecognize();
print "Loading Weights List..";
wl=weightsList.weightsList();
print "Weights List Complete!";
data= weightsList.outputList();
#msetrained = weightsList.trained_mse_list();
C.numInput(int(data[0]));
C.numHidden(int(data[1]));
C.numOutput(int(data[2]));
C.TotalImages(int(data[4]));
C.avHeight(int(data[5]));
C.avWidth(int(data[6]));
vectorOutput = C.MakeMatrix(C.get_TotalImages(),C.get_numOutput());
for i in range(C.get_TotalImages()):
    decnum=i+1; 
    out =evaluate.dec_to_bin(decnum);
    tempidx = C.get_numOutput()-len(str(out));
    for j in range(0,len(str(out))):
        num = str(out);
        vectorOutput[i][tempidx+j]=int(num[j]);
C.output(vectorOutput);

img=C.MakeMatrix(1,C.get_avHeight()*C.get_avWidth());
#result,avHeight,avWidth = new_image.imagedirectory("F:\\Google Drive\\Proyecto Tesis Licenciatura\\All Patterns\\AgencyFB\\Prob20\\First Test",".jpg");
result,avHeight,avWidth = new_image.imagedirectory("C:\\Users\\Jacob\\Pictures\\Google Photos Backup",".bmp");



for i in range(C.get_TotalImages()):
    img=new_image.matrixgrayoutimage(result[i],C.get_avHeight(),C.get_avWidth(),img,0);
    C.MatrixInput(img);
    evaluate.CalculateOutput(wl,i);


