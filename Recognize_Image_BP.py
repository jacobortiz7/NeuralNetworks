from PIL import Image
import cv2
import math
import numpy as np 
import os
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
    def outputList(self):
        inp = open ("F:\\Google Drive\\Proyecto Tesis Licenciatura\\All Patterns\\AgencyFB\\arquitecture_nn.txt","r")
        #read line into array 
        data = []
        for line in inp.readlines():
            # loop over the elemets, split by whitespace
            for i in line.split():
                data.append(i)
        return data; 
    def weightsList(self,file):
        arr = []
        arr2= []
        inp = open (file,"r")
        finlinea=False;
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
                    if (j<(len(i)-1) and i[j].__str__()=="]" and (i[j+1].__str__()== "," or i[j+1].__str__()== "]")):
                        finlinea=True;
                list.append(float(temp))
                if(finlinea==True):
                    arr.append(list)
                    finlinea=False
                    list = []
                    if(i[j-2].__str__()=="]" and i[j-1].__str__()=="]" and (i[j].__str__()== "," or i[j].__str__()== "]")):
                        arr2.append(arr);
                        arr = []
            c=c+1;
            print "Weights loaded from image #",c;
            
        return arr2;  

class Forward():
    def sigmoid(x):
        return (1 / (1 + math.exp(-x)))

    def dec_to_bin(self,x):
        return int(bin(x)[2:])

    def bin_vector(self,x):
        matrix = []
        for i in range(0,len(str(x))):
            num = str(x);
            matrix.append(int(num[i]));
        return matrix

    def CalculateOutput(self,wi,wo,inputs,image):
        ao = C.MakeMatrix(1,C.get_numOutput())
        ah = C.MakeMatrix(1,C.get_numHidden())
        matrixOut = C.get_output();
        maxValue=1;
        MatchedHigh="?";
        MatchedLow="?";
        mseList=[];
        for images in range (C.get_TotalImages()):
            trout = matrixOut[images];
            for j in range(C.get_numHidden()):
                sum = 0.0
                for i in range(C.get_numInput()):
                    sum +=( inputs[0][i] * wi[images][i][j] )
                    ah[0][j] = (1 / (1 + math.exp(-sum)))
        
            for k in range(C.get_numOutput()):
                sum = 0.0
                for j in range(C.get_numHidden()):        
                    sum +=( ah[0][j] * wo[images][j][k] )
                    ao[0][k] = (1 / (1 + math.exp(-sum)))
            mse = ((np.array([trout])-np.array(ao))**2).mean(axis=None);
            mseList.append(mse);
            if(float(mse)<maxValue):
                MatchedLow=MatchedHigh;
                OutputValueHigh=maxValue;
                maxValue=float('%.17f'%mse);
                MatchedHigh=trout;
                OutputValueHight = maxValue;

        print("Matched High:"+str(MatchedHigh)+'\n');
        #print("Matched Low:"+str(MatchedLow)+'\n');
        inp = open('F:\\Google Drive\\Proyecto Tesis Licenciatura\\All Patterns\\AgencyFB\\Prob30\\Third Test\\BP Results\\recognitionResultsBP.txt', 'a+')
        #inp = open('F:\\Google Drive\\Proyecto Tesis Licenciatura\\All Patterns\\AgencyFB\\recognitionResultsBP.txt', 'a+')
        #inp.write(str('[Output Desired]')+str('[Final output]')+str('[MSE]')+'\n');
        inp.write(str(MatchedHigh)+'\n');
        #inp.write("Matched Low:"+str(MatchedLow)+'\n');
        inp.close()
        inp = open('F:\\Google Drive\\Proyecto Tesis Licenciatura\\All Patterns\\AgencyFB\\Prob30\\Third Test\\BP Results\\mseRecognitionBP.txt', 'a+')
        #inp = open('F:\\Google Drive\\Proyecto Tesis Licenciatura\\All Patterns\\AgencyFB\\mseRecognitionBP.txt', 'a+')
        inp.write(str(mseList[image])+'\n');
        inp.close()
        return ao
new_image = IMAGE();
weightsList = ReadWeights();
C = Const();
P=Forward();
wi=weightsList.weightsList("F:\\Google Drive\\Proyecto Tesis Licenciatura\\All Patterns\\AgencyFB\\weights_wi.txt");
wo=weightsList.weightsList("F:\\Google Drive\\Proyecto Tesis Licenciatura\\All Patterns\\AgencyFB\\weights_wo.txt");
data= weightsList.outputList();
C.numInput(int(data[0]));
C.numHidden(int(data[1]));
C.numOutput(int(data[2]));
C.TotalImages(int(data[4]));
C.avHeight(int(data[5]));
C.avWidth(int(data[6]));
vectorOutput = C.MakeMatrix(C.get_TotalImages(),C.get_numOutput());
for i in range(C.get_TotalImages()):
    decnum=i+1; 
    out =P.dec_to_bin(decnum);
    tempidx = C.get_numOutput()-len(str(out));
    for j in range(0,len(str(out))):
        num = str(out);
        vectorOutput[i][tempidx+j]=int(num[j]);
C.output(vectorOutput);

result,avHeight,avWidth = new_image.imagedirectory("F:\\Google Drive\\Proyecto Tesis Licenciatura\\All Patterns\\AgencyFB\\Prob30\\Third Test",".jpg");
#result,avHeight,avWidth = new_image.imagedirectory("F:\\Google Drive\\Proyecto Tesis Licenciatura\\All Patterns\\AgencyFB",".png");
for i in range(C.get_TotalImages()):
    img=C.MakeMatrix(1,C.get_avHeight()*C.get_avWidth());
    img=new_image.matrixgrayoutimage(result[i],C.get_avHeight(),C.get_avWidth(),img,0);
    P.CalculateOutput(wi,wo,img,i);
     


