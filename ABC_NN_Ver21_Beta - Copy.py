"""ABC_NN.py: This file has implemented an Artificial Neural Network training with Artificial Bee Colony."""
__author__ = 'Jacob Ortiz Escobedo'
__credits__ = ["PhD Sara Elena Garza Villareal"]
__license__ = "GPL"
__version__ = "1.23"
__swenginner__ = "Jacob Ortiz Escobedo"
__email__ = "jacob.ortiz@gmail.com"
__status__ = "Production"
import numpy as np
import random
import math
import cv2
import os
import sys
import time
from PIL import Image

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

class Arguments(object):
    def imagedirectory(self,path,filetype):
        list=[]
        avHeight=0;
        avWidth=0;
        print "DIRECTORY: ",path
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
  

  #def createNeuralNetwork(self,numInput,numHidden,numOutput):
      #self.numInputs = C.MakeMatrix(1,numInput);
      #self.ihWeights = C.MakeMatrix(numInput,numHidden);
      #self.ihBiases = C.MakeMatrix(1,numHidden);
      #self.hoWeights = C.MakeMatrix(numHidden,numOutput);
      #self.hoBiases = C.MakeMatrix(1,numOutput);
      #self.outputs = C.MakeMatrix(1,numOutput);
  def Directory(self,directory):
      self.path = directory;

  def get_Directory(self):
      return self.path;
    
  def NumberColonySize(self,cs):
      self.colonysize = cs;
  
  def get_NumberColonySize(self):
      return self.colonysize;

  def TotalImages(self,tot):
      self.total = tot;

  def get_TotalImages(self):
      return self.total

  def MatrixInput(self,matrix):
      self.matrix = matrix;
  
  def get_MatrixInput(self):
      return self.matrix;

  def MaxCycle(self,max):
      self.max = max;

  def get_MaxCycle(self):
      return self.max;

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
   
  def FoodNumber(self):
      return C.get_NumberColonySize()/2;
    
  def CreateVectorUB(self):
      UB=[]
      rows=1;  
      columns=C.get_D();
      for i in range(rows):
        UB.append([15.2]*columns);
      return UB
                
  def CreateVectorLB(self):
      LB=[]
      rows=1;
      columns=C.get_D();
      for i in range(rows):
        LB.append([-15.2]*columns);
      return LB
    
  def Limit(self):
      return Const.FoodNumber(self)*C.get_D();

  def RunTime(self):
      return 1;

  def Get_Trial(self):
    global trial
    return trial

  def GlobalMins(self):
      Zeros=[]
      rows=1;
      columns=Const.RunTime(self);
      for i in range(rows):
        Zeros.append([0]*columns);
      return Zeros

  def RepMat(self,Matrix,FoodNumber):
    #w, h = 8, 5;
    #Matrix = [[0 for x in range(w)] for y in range(h)] 
    RepMatrix=[]
    #rows,columns = Matrix.shape;
    rows = len (Matrix)
    columns = len ([Matrix[0]])
    for i in range(rows):
      RepMatrix.append([Matrix]*FoodNumber);
    return RepMatrix

  def MakeMatrix ( self,I, J, fill=0):
    m = []
    for i in range(I):
      m.append([fill]*J)
    return m
  
  def FillMatrixFoods(self,UB,LB): 
    Foods=[]
    for h in range(0,C.get_TotalImages()):
      matrix = Const.MakeMatrix(self,Const.FoodNumber(self),C.get_D());
      for i in range ( len (matrix) ): 
        for j in range ( len (matrix[0]) ):
          r= (float)(random.random())*(float)(32767)/((float)(32767)+(float)(1))
          matrix[i][j] = r*(UB[0][0]-LB[0][0])+LB[0][0] #random.uniform(LB[0][0],UB[0][0]); 
     
      Foods.append(matrix);
    return Foods  

class NNABC(object):
  def GetColumnMatrix(self,Foods,rows,idx_j):
    for i in range (rows):
      ColumVector = []
      ColumVector.append([Foods[i][idx_j]]);
    return ColumVector

  def GetVectorElements(self,vector,a,b):
    rows = 1;
    columns = b+1-a;
    x=[]
    x= Const.MakeMatrix(self,rows,columns);
    for i in range ( len (x) ):
      for j in range ( len (x[0]) ):
        x[i][j] = vector[a+j-1];
        #a=a+1;
    return x  
     
  #Sigmoid function
  def Sigmoid(self,x): 
      return 1 / (1 + np.exp(-x))
    
  #Hyperbolic tangent function
  def sigmoid (self,x):
    return math.tanh(x)

  def ReadVector(self,vector):
    rows=len(vector);
    columns = len(vector[0]);
    m=Const.MakeMatrix(self,rows,columns)  
    for k in range(rows):
      for j in range(columns):
        m[k][j]=NNABC.Sigmoid(self,vector[k][j]); 
    return m
        
  def MemorizeBestSource(self,GlobalParams,GlobalMin,ObjVal,Foods):
    #%/*The best food source is memorized*/
    for l in range(C.get_TotalImages()):
      for m in range (Const.FoodNumber(self)):
        if(ObjVal[l][m]<GlobalMin[l]):
          print "Latest GlobalMin found:",GlobalMin[l],",New GlobalMin found:",ObjVal[l][m]
          GlobalMin[l]=ObjVal[l][m];
          for n in range (C.get_D()):
            GlobalParams[l][n]=Foods[l][m][n];
    print "GLOBAL MSE:",np.mean(GlobalMin);    
    return (GlobalParams,GlobalMin)


  def ReplaceBestSource(self,GlobalParams,GlobalMin,ObjVal,Foods):
    for l in range(C.get_TotalImages()):
      for m in range (Const.FoodNumber(self)):
        if(ObjVal[l][m]==GlobalMin[l]):
          #print "Latest GlobalMin found:",GlobalMin,",New GlobalMin found:",ObjVal[l][m]
          GlobalMin[l]=ObjVal[l][m];
          for n in range (C.get_D()):
            GlobalParams[l][n]=Foods[l][m][n];
          break;
    print "GLOBAL MSE:",np.mean(GlobalMin);
    return (GlobalParams,GlobalMin)

  def CalculateFitnessNum(self,m):
    fFitness=0;
    if(m>=0):
      fFitness=float(1/(m+1));
    else:
      fFitness=1+math.fabs(m)
    return fFitness 
    
  def CalculateFitness(self,m):
    fFitness=[]
    rows=len(m);
    columns = len(m[0]);
    fFitness = Const.MakeMatrix(self,rows,columns);
    for k in range(rows):
      for j in range(columns):
        if(m[k][j]>=0):
          div=m[k][j];
          fFitness[k][j]=float(1/(div+1));
        else:
          fFitness[k][j]=1+math.fabs(m[k][j])
    return fFitness 
    
  def SendEmployedBees(self,f,Foods,LB,UB,Fitness,ObjVal,trial):
    sol = []
    for ff in range(C.get_TotalImages()):
      sol.append(C.MakeMatrix(len(Foods[0]),C.get_D()));
    for k in range(Const.FoodNumber(self)):
      #The parameter to be changed is determined randomly
      r = (random.random()*32767/(32767+1));
      Param2Change=int(np.fix(r*C.get_D()));
      #A randomly chosen solution is used in producing a mutant solution of the solution i
      r = (random.random()*32767/(32767+1));
      neighbour=int(np.fix(r*(Const.FoodNumber(self))));
      #Randomly selected solution must be different from the solution i
      while(neighbour==k):
        r = (random.random()*32767/(32767+1));
        neighbour=int(np.fix(r*(Const.FoodNumber(self))));
      for col in range (len(Foods[f][k])):
        sol[f][k][col] = Foods[f][k][col]
      #v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij})#
      FitnessSol=0; 
      
      sol[f][k][Param2Change] = (Foods[f][k][Param2Change]+(Foods[f][k][Param2Change]-Foods[f][neighbour][Param2Change])*random.uniform(-1,1)*2);
      #/*if generated parameter value is out of boundaries, it is shifted onto the boundaries*/
      if (sol[f][k][Param2Change]<LB[0][Param2Change]):
        sol[f][k][Param2Change]=LB[0][Param2Change];
      if(sol[f][k][Param2Change]>UB[0][Param2Change]):
        sol[f][k][Param2Change]=UB[0][Param2Change];
      #evaluate new solution
      ObjValSol=NNABC.Train(self,[sol[f][k]],f);
      FitnessSol=NNABC.CalculateFitnessNum(self,ObjValSol);
      #a greedy selection is applied between the current solution i and its mutant
      if (FitnessSol>Fitness[f][k]): #If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i*/
        print "Foods[",f,"][",k,"][",Param2Change,"]",Foods[f][k][Param2Change],"sol[",f,"][",k,"][",Param2Change,"]",sol[f][k][Param2Change]
        print FitnessSol,">",Fitness[f][k],": ",FitnessSol>Fitness[f][k];
        print "ObjVal[",f,"][",k,"]",ObjVal[f][k],"ObjValSol",ObjValSol
        for j in range(C.get_D()): 
          Foods[f][k][j]=sol[f][k][j];
        Fitness[f][k] = FitnessSol
        trial[f][k]=0
        ObjVal[f][k]=ObjValSol;
      else:
        trial[f][k]=trial[f][k]+1; #if the solution i can not be improved, increase its trial counter*/
    return (Fitness,Foods,ObjVal,trial)

  def CalculateProbabilitiesMatrix(self,Fitness,Foods,ObjVal,trial):
    rows=len(Fitness);
    columns = len(Fitness[0]);
    prob=[]
    prob = Const.MakeMatrix(self,rows,columns);
    maximum=0;
    for i in range(0,rows):
      for j in range(0,columns):
        if(Fitness[i][j]>maximum):
          maximum = Fitness[i][j]
      
    for i in range(0,rows):
      for j in range(0,columns):
        prob[i][j]=(0.9*float(np.array(Fitness[i][j])/np.array(maximum)))+0.1;
    return prob  
      
  def SendOnlookerBees(self,f,prob,Foods,LB,UB,Fitness,ObjVal,trial):
    sol = []
    for ff in range(C.get_TotalImages()):
      sol.append(C.MakeMatrix(len(Foods[0]),C.get_D()));
    
    k=0;
    t=0;

    while (t<=Const.FoodNumber(self)):
      r = (random.random()*32767/(32767+1));
      if(r<prob[f][k]):
        t=t+1
        #%/*The parameter to be changed is determined randomly*/
        r = (random.random()*32767/(32767+1));
        Param2Change=int(np.fix(r*C.get_D()));
        #%/*A randomly chosen solution is used in producing a mutant solution of the solution i*/
        r = (random.random()*32767/(32767+1));
        neighbour=int(np.fix(r*(Const.FoodNumber(self))));
        #%/*Randomly selected solution must be different from the solution i*/        
        while(neighbour==k):
          r = (random.random()*32767/(32767+1));
          neighbour=int(np.fix(r*(Const.FoodNumber(self))));
        for col in range (len(Foods[f][k])):
          sol[f][k][col] = Foods[f][k][col]
        #%  /*v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) */
        r = (random.random()*32767/(32767+1));
        sol[f][k][Param2Change] =  Foods[f][k][Param2Change]+(Foods[f][k][Param2Change]-Foods[f][neighbour][Param2Change])*(r-0.5)*2;
        #%  /*if generated parameter value is out of boundaries, it is shifted onto the boundaries*/
        if (sol[f][k][Param2Change]<LB[0][Param2Change]):
          sol[f][k][Param2Change]=LB[0][Param2Change];
        if(sol[f][k][Param2Change]>UB[0][Param2Change]):
          sol[f][k][Param2Change]=UB[0][Param2Change];
        #%evaluate new solution
        ObjValSol=NNABC.Train(self,[sol[f][k]],f);
        FitnessSol=NNABC.CalculateFitnessNum(self,ObjValSol); 
        #% /*a greedy selection is applied between the current solution i and its mutant*/
        if (FitnessSol>Fitness[f][k]):
          print FitnessSol,">",Fitness[f][k],": ",FitnessSol>Fitness[f][k];
          print "Foods[",f,"][",k,"][",Param2Change,"]",Foods[f][k][Param2Change],"sol[",f,"][",k,"][",Param2Change,"]",sol[f][k][Param2Change]
          print "ObjVal[",f,"][",k,"]",ObjVal[f][k],"ObjValSol",ObjValSol
          for j in range(C.get_D()):
            Foods[f][k][j]= sol[f][k][j];
          Fitness[f][k] = FitnessSol
          trial[f][k]=0
          ObjVal[f][k]=ObjValSol;
        else:
          trial[f][k]=trial[f][k]+1; #%/*if the solution i can not be improved, increase its trial counter*/
          k=k+1;
          if(k==Const.FoodNumber(self)):
            k=0;
    return (Fitness,Foods,ObjVal,trial)         

  def SendScoutBees(self,f, trial,NM,LB,Range,Lower,Fitness,ObjVal,Foods):
    #%/*determine the food sources whose trial counter exceeds the "limit" value. 
    #%In Basic ABC, only one scout is allowed to occur in each cycle*/
    FitnessSol=[]
    rows=len(trial);
    columns = len(trial[0]);
    maximum=0;
    for i in range(0,rows):
      for j in range(0,columns):
        if(trial[i][j]>maximum):
          row = i
          colum =j
          maximum = trial[i][j]

    if(maximum>=Const.Limit(self)):
      sol= Const.MakeMatrix(self,len(NM),len(NM[0]));
      for r in range(len(NM)):
          for c in range(len(NM[0])):
            sol[r][c] = NM[r][c]*random.uniform(-C.get_D(),C.get_D());

      ObjValSol=NNABC.Train(self,sol,f);
      FitSol=NNABC.CalculateFitnessNum(self,ObjValSol);
      FitnessSol.append(FitSol);
      Foods[f][colum]=sol[0];
      Fitness[f][colum] = FitSol;
      ObjVal[f][colum]=ObjValSol;
            
    return (ObjVal,Fitness,Foods)

  def Train(self,Foods,img):
    rows=1;
    columns = len(Foods[0]);
    ObjVal=0
    matrixIn = C.get_MatrixInput();
    matrixOut = C.get_output();
    for ni in range (rows):
      x= Foods[ni]
      trout = matrixOut[img];
      self.trin= matrixIn[img];
      iw = NNABC.GetVectorElements(self,x,1,C.get_numInput()*C.get_numHidden()); 
      b1 = NNABC.GetVectorElements(self,x,C.get_numInput()*C.get_numHidden()+1,C.get_numInput()*C.get_numHidden()+C.get_numHidden()); 
      lw = NNABC.GetVectorElements(self,x,C.get_numInput()*C.get_numHidden()+C.get_numHidden()+1,C.get_numInput()*C.get_numHidden()+C.get_numHidden()+C.get_numHidden()*C.get_numOutput());
      b2 = NNABC.GetVectorElements(self,x,C.get_numInput()*C.get_numHidden()+C.get_numHidden()+C.get_numHidden()*C.get_numOutput()+1,C.get_numInput()*C.get_numHidden()+C.get_numHidden()+C.get_numHidden()*C.get_numOutput()+C.get_numOutput());

      ihSums = C.MakeMatrix(1,C.get_numHidden());
      
      c=0;
      for i in range(0,C.get_numHidden()):
        for j in range(0, C.get_numInput()):
            ihSums[0][i] += self.trin[j]*iw[0][c];
            c=c+1; 
            
      for k in range(0,C.get_numHidden()):
        ihSums[0][k] += b1[0][k];  
              
      ihOutputs = NNABC.ReadVector(self,ihSums);
      hoSums = C.MakeMatrix(1,C.get_numOutput());


      c=0;
      for l in range(0,C.get_numOutput()):
        for m in range(0,C.get_numHidden()):  
            hoSums[0][l] += ihOutputs[0][m]*lw[0][c];
            c=c+1;
       
      for n in range(0,C.get_numOutput()):
        hoSums[0][n] +=  b2[0][n];
        

      y = NNABC.ReadVector(self,hoSums);         
      mse = ((np.array(trout)-np.array(y))**2).mean(axis=None);
      ObjVal = mse;
    return ObjVal  

  def TrainABC(self,Foods):
    rows=len(Foods[0]);
    columns = len(Foods[0][0]);
    x=[]
    #ObjVal=[]
    matrixIn = C.get_MatrixInput();
    matrixOut = C.get_output();
    ObjVal= C.MakeMatrix(len(matrixIn),rows);
    for ni in range(len(matrixIn)):
        for ir in range (rows):
            x= Foods[ni][ir]
            trout = matrixOut[ni];
            self.trin = matrixIn[ni];

            iw = NNABC.GetVectorElements(self,x,1,C.get_numInput()*C.get_numHidden()); 
            b1 = NNABC.GetVectorElements(self,x,C.get_numInput()*C.get_numHidden()+1,C.get_numInput()*C.get_numHidden()+C.get_numHidden()); 
            lw = NNABC.GetVectorElements(self,x,C.get_numInput()*C.get_numHidden()+C.get_numHidden()+1,C.get_numInput()*C.get_numHidden()+C.get_numHidden()+C.get_numHidden()*C.get_numOutput());
            b2 = NNABC.GetVectorElements(self,x,C.get_numInput()*C.get_numHidden()+C.get_numHidden()+C.get_numHidden()*C.get_numOutput()+1,C.get_numInput()*C.get_numHidden()+C.get_numHidden()+C.get_numHidden()*C.get_numOutput()+C.get_numOutput());
        
            ihSums = C.MakeMatrix(1,C.get_numHidden());
            
            c=0;
            for i in range(0,C.get_numHidden()):
                for j in range(0, C.get_numInput()):
                    ihSums[0][i] += self.trin[j]*iw[0][c];
                    c=c+1; 

                            
            for k in range(0,C.get_numHidden()):
                ihSums[0][k] += b1[0][k];  
                          
            ihOutputs = NNABC.ReadVector(self,ihSums);
            hoSums = C.MakeMatrix(1,C.get_numOutput());
        
        
            c=0;
            for l in range(0,C.get_numOutput()):
                for m in range(0,C.get_numHidden()):    
                    hoSums[0][l] += ihOutputs[0][m]*lw[0][c];
                    c=c+1;
               
            for n in range(0,C.get_numOutput()):
                hoSums[0][n] +=  b2[0][n];
                
        
            y = NNABC.ReadVector(self,hoSums);         
                  
            mse = ((np.array([trout])-np.array(y))**2).mean(axis=None);
            ObjVal[ni][ir]=mse;
    return ObjVal  

  def CalculateOutput(self,Foods):
    rows=len(Foods);
    columns = len(Foods[0]);
    x=[]
    ObjVal=[]
    matrixIn = C.get_MatrixInput();
    matrixOut = C.get_output();
    for ni in range(rows):
        x= Foods[ni]
        trout = matrixOut[ni];
        self.trin = matrixIn[ni];
        iw = NNABC.GetVectorElements(self,x,1,C.get_numInput()*C.get_numHidden()); 
        b1 = NNABC.GetVectorElements(self,x,C.get_numInput()*C.get_numHidden()+1,C.get_numInput()*C.get_numHidden()+C.get_numHidden()); 
        lw = NNABC.GetVectorElements(self,x,C.get_numInput()*C.get_numHidden()+C.get_numHidden()+1,C.get_numInput()*C.get_numHidden()+C.get_numHidden()+C.get_numHidden()*C.get_numOutput());
        b2 = NNABC.GetVectorElements(self,x,C.get_numInput()*C.get_numHidden()+C.get_numHidden()+C.get_numHidden()*C.get_numOutput()+1,C.get_numInput()*C.get_numHidden()+C.get_numHidden()+C.get_numHidden()*C.get_numOutput()+C.get_numOutput());
        
        ihSums = C.MakeMatrix(1,C.get_numHidden());
        
        c=0;
        for i in range(0,C.get_numHidden()):
            for j in range(0, C.get_numInput()):
                ihSums[0][i] += self.trin[j]*iw[0][c];
                c=c+1; 
                    
        for k in range(0,C.get_numHidden()):
            ihSums[0][k] += b1[0][k];  
                      
        ihOutputs = NNABC.ReadVector(self,ihSums);
        hoSums = C.MakeMatrix(1,C.get_numOutput());
        
        
        c=0;
        for l in range(0,C.get_numOutput()):
            for m in range(0,C.get_numHidden()):
                hoSums[0][l] += ihOutputs[0][m]*lw[0][c];
                c=c+1;
              
        for n in range(0,C.get_numOutput()):
            hoSums[0][n] += b2[0][n];
                
        outputs = NNABC.ReadVector(self,hoSums);         
        mse = ((np.array([trout])-np.array(outputs))**2).mean(axis=None);
      
          
        inp = open(C.get_Directory()+'\\'+'outputs.txt', 'a+')
        #inp.write(str('[Output Desired]')+str('[Final output]')+str('[MSE]')+'\n');
        inp.write(str(trout)+str(outputs)+str(x)+str(mse)+'\n');
        inp.close()

        inp = open(C.get_Directory()+'\\'+'weights.txt', 'a+')
        #inp.write(str('[Output Desired]')+str('[Final output]')+str('[MSE]')+'\n');
        inp.write(str(x)+'\n');
        inp.close()


        inp = open(C.get_Directory()+'\\'+'mse_trained.txt', 'a+')
        #inp.write(str('[Output Desired]')+str('[Final output]')+str('[MSE]')+'\n');
        inp.write(str(mse)+'\n');
        inp.close()
        #return outputs,mse;
       
    
class ABC(Const,NNABC):
  def CycleFor(self):
    ObjVal=[]
    MatrixInputs=[]
    LastWeights=[]
    for i in range(0,Const.RunTime(self)):
      UB= Const.CreateVectorUB(self);
      LB= Const.CreateVectorLB(self);
      NM= Const.MakeMatrix(self,len(UB),len(UB[0]));
      for r in range(len(UB)):
          for c in range(len(UB[0])):
            NM[r][c] = UB[r][c]-LB[r][c]
      Range= Const.RepMat(self,NM,Const.FoodNumber(self));
      Lower = Const.RepMat(self,LB,Const.FoodNumber(self));
      print "Generating Initial Foods Stared..."
      Foods=Const.FillMatrixFoods(self,UB,LB);
      print "Generating Initial Foods Finished..."
      print "Initial Training Stared..."
      ObjVal=NNABC.TrainABC(self,Foods);
      print "Initial Training Finished..."
      print "Initial Fitness Stared..."
      Fitness=NNABC.CalculateFitness(self,ObjVal);
      print "Initial Fitness Finished..."
      trial = Const.MakeMatrix(self,C.get_TotalImages(),Const.FoodNumber(self));
      

      print "Searching Initial Global Values..."
      GlobalMin = C.MakeMatrix(C.get_TotalImages(),1);
      for gmin in range(C.get_TotalImages()):
          for row in range(1):
            GlobalMin[gmin][row]=np.min(ObjVal[gmin]);
      print GlobalMin;
      print "All Initial Global Values Saved in memory..."
      GlobalParams = Const.MakeMatrix(self,C.get_TotalImages(),C.get_D());
      print "Replacing GlobalParams through GlobalMins Founded with Initial Foods Generated..."
      GlobalParams,GlobalMin= NNABC.ReplaceBestSource(self,GlobalParams,GlobalMin,ObjVal,Foods);
      print "Replacing GlobalParams Finished..."
     # while ((np.mean(GlobalMin)>3.06096648272e-09)): #and (iteration<C.get_MaxCycle())):
      for picnumber in range(C.get_TotalImages()):
          iteration = 0;
          #while (iteration<C.get_MaxCycle()):
          while (GlobalMin[picnumber]>0): #and (iteration<C.get_MaxCycle())):
            print "PICTURE",picnumber,"ITERATION",iteration
            iteration = iteration +1
            print "Sending Employed Bees..."
            Fitness,Foods,ObjVal,trial = NNABC.SendEmployedBees(self,picnumber,Foods,LB,UB,Fitness,ObjVal,trial);
            print "Calculating probabilities..."
            prob = NNABC.CalculateProbabilitiesMatrix(self,Fitness,Foods,ObjVal,trial);
            print "Sending Onlooker Bees..."
            Fitness,Foods,ObjVal,trial = NNABC.SendOnlookerBees(self,picnumber,prob,Foods,LB,UB,Fitness,ObjVal,trial);
            print "Memorizing Best Source..."
            GlobalParams,GlobalMin = NNABC.MemorizeBestSource(self,GlobalParams,GlobalMin,ObjVal,Foods);
            print "Sending Scout Bees..."
            ObjVal,Fitness,Foods = NNABC.SendScoutBees(self,picnumber,trial,NM,LB,Range,Lower,Fitness,ObjVal,Foods);

    NNABC.CalculateOutput(self,GlobalParams);
  def dec_to_bin(self,x):
    return int(bin(x)[2:])

  def bin_vector(self,x):
    matrix = []
    for i in range(0,len(str(x))):
        num = str(x);
        matrix.append(int(num[i]));
    return matrix

new_image = IMAGE();
avHeight=0;
avWidth=0;
image_list = Arguments();
#result,avHeight,avWidth = image_list.

#(sys.argv[1],sys.argv[2]);
result,avHeight,avWidth = image_list.imagedirectory("C:\\Users\\Jacob\\Pictures\\Google Photos Backup",".bmp");
C=Const();
C.Directory("C:\\Users\\Jacob\\Pictures\\Google Photos Backup");
C.TotalImages((int)(len(result)));
FullMatrix=C.MakeMatrix(len(result),avHeight*avWidth);
#print FullMatrix
ClassABC=ABC(); 
total_lenght_outputs=len(str(ClassABC.dec_to_bin(len(result))));
vectorOutput = C.MakeMatrix(len(result),total_lenght_outputs);
for i in range(0,len(result)):
    FullMatrix = new_image.matrixgrayoutimage(result[i],avHeight,avWidth,FullMatrix,i);
    decnum=i+1;
    out =ClassABC.dec_to_bin(decnum);
    temp=total_lenght_outputs;
    tempidx = total_lenght_outputs-len(str(out));
    for j in range(0,len(str(out))):
        num = str(out);
        vectorOutput[i][tempidx+j]=int(num[j]);

C.output(vectorOutput);

print len(FullMatrix),len(FullMatrix[0])
C.MatrixInput(FullMatrix);
C.numInput(len(FullMatrix[0]));
C.numOutput(total_lenght_outputs);
hidden=((int)(math.ceil(math.sqrt((C.get_numInput()*C.get_numOutput()))))); 
#hidden=((int)(math.ceil((2.0/3.0)*(C.get_numInput()*C.get_numOutput())))); 
C.numHidden(hidden);    
C.MaxCycle(1000);
C.NumberColonySize(50);
C.D(C.get_numInput(),C.get_numHidden(),C.get_numOutput());
print "Total inputs:", C.get_numInput();
print "Hidden Neurons number:", C.get_numHidden();
print "Total outputs:",C.get_numOutput();
print "Total Weights of Neural Network:", C.get_D();
print "Total Number Colony Size:", C.get_NumberColonySize();
print "Maximum Cycles:", C.get_MaxCycle();
print "Total Images Found:",C.get_TotalImages();
#print "Name of image in progress:",result[i];
#print "Number of image in progress:",index;
inp = open(C.get_Directory()+'\\'+'arquitecture_nn.txt', 'a+')
inp.write(str(C.get_numInput())+'\n');
inp.write(str(C.get_numHidden())+'\n');
inp.write(str(C.get_numOutput())+'\n');
inp.write(str(C.get_D())+'\n');
inp.write(str(C.get_TotalImages())+'\n');
inp.write(str(avHeight)+'\n');
inp.write(str(avWidth)+'\n');

inp.close();

inp = open(C.get_Directory()+'\\'+'arquitecture_nn_output.txt', 'a+')
inp.write(str(vectorOutput)+'\n');
inp.close();

ClassABC.CycleFor();
        


